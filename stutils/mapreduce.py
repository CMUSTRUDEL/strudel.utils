
"""
Alternative approaches:

`asyncio`

    - Python 3.4+. Unfortunately, I have to support Python 2, mainly because of
        OSCAR project compatibility

`concurrent.futures`

    - available since Python 3.2, but there is a backport
        https://pypi.org/project/futures/
    - backport follows pretty much the same pattern but still needs a wrapper
        to support pandas objects. Native version (Python 3.2+) might give
        some performance advantages


native `ThreadPool` (`from multiprocessing.pool import ThreadPool`):

    - 30..50% slower than this implementation
        (see test.TestMapReduce.test_native_threadpool)
    - (minor) doesn't support pandas objects

"""

import pandas as pd
import six

import logging
import multiprocessing
import threading
import time


CPU_COUNT = multiprocessing.cpu_count()


class ThreadPool(object):
    """ A slightly more performant replacement for native ThreadPool.

    States:
        started, stopped
        False, False: just created, threads don't run yet
            probably building up queue of tasks in the main thread
        False, True: terminated before start. Shouldn't happen in the wild.
            pretty useless state, identical to False, False
        True, False: started and processing
        True, True: finished processing or terminated

    """
    _threads = None
    queue = None
    started = False
    stopped = False
    callback_semaphore = None

    def __init__(self, n_workers=None):
        # the only reason to use threadpool in Python is IO (because of GIL)
        # so, we're not really limited with CPU and twice as many threads
        # is usually fine as a default
        self.n = n_workers or CPU_COUNT * 2
        self.queue = six.moves.queue.Queue()
        self.callback_semaphore = threading.Lock()

    def start(self):
        if self.started:
            logging.warning("The pool is already started")
            return

        def worker():
            while not self.stopped and (self.started or not self.queue.empty()):
                # self.stopped: Terminate immediately
                # self.started, queue is not empty: normal processing
                # self.started, queue is empty: waiting for main thread
                #   to submit more tasks
                # not self.started, queue is not empty: shutting down gracefully
                # not self.started, queue is empty: done, exit

                try:
                    func, args, kwargs, callback = self.queue.get(False)
                except six.moves.queue.Empty:
                    time.sleep(0.1)
                    continue
                else:
                    logging.debug("Got new data")

                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    logging.exception(e)
                else:
                    logging.debug(
                        "Processed data: %s -> %s", str(args), str(result))
                    self.callback_semaphore.acquire()
                    try:
                        callback(result)
                    except Exception as e:
                        logging.exception(e)
                    finally:
                        self.callback_semaphore.release()

        self._threads = [threading.Thread(target=worker) for _ in range(self.n)]
        self.started = True
        [t.start() for t in self._threads]

    def submit(self, func, *args, **kwargs):
        # submit is executed from the main thread and expected to be synchronous
        callback = None
        if 'callback' in kwargs:
            callback = kwargs.pop('callback')
            assert callable(callback), "Callback must be callable"

        self.queue.put((func, args, kwargs, callback))

        if not self.started:
            self.start()

    def shutdown(self):
        """Wait for all threads to complete"""
        # cleanup
        self.started = False
        try:
            # nice way of doing things - let's wait until all items
            # in the queue are processed
            for t in self._threads:
                t.join()
        finally:
            # Emergency brake - if a KeyboardInterrupt is raised,
            # threads will finish processing current task and exit
            self.stopped = True

    def __del__(self):
        self.shutdown()


def map(func, data, num_workers=None):
    # type: (callable, Iterable, Optional[int]) -> Iterable
    """ Map an iterable using multithreading
    >>> s = pd.Series(range(120, 0, -1))
    >>> s2 = map(lambda i, x: x ** 3.75, s)
    >>> isinstance(s2, type(s))
    True
    >>> len(s) == len(s2)
    True
    >>> (s2 == s.map(lambda x: x ** 3.75)).all()
    True
    >>> s = list(range(120, 0, -1))
    >>> s2 = map(lambda i, x: x ** 3.75, s)
    >>> isinstance(s2, type(s))
    True
    >>> len(s) == len(s2)
    True
    >>> all(x ** 3.75 == s2[i] for i, x in enumerate(s))
    True
    >>> s = dict(enumerate(range(120, 0, -1)))
    >>> s2 = map(lambda i, x: x ** 3.75, s)
    >>> isinstance(s2, type(s))
    True
    >>> len(s) == len(s2)
    True
    >>> all(x ** 3.75 == s2[i] for i, x in s.items())
    True
    """
    backend = ThreadPool(n_workers=num_workers)
    iterable = None
    # pd.Series didn't have .items() until pandas 0.21,
    # so iteritems for older versions
    for method in ('iterrows', 'iteritems', 'items'):
        if hasattr(data, method):
            iterable = getattr(data, method)()
            break
    if iterable is None:
        iterable = enumerate(data)

    mapped = {}

    def collect(key):
        def process(res):
            mapped[key] = res
        return process

    for key, value in iterable:
        backend.submit(func, key, value, callback=collect(key))

    backend.shutdown()

    if isinstance(data, pd.DataFrame):
        return pd.DataFrame.from_dict(
            mapped, orient='index').reindex(data.index)
    elif isinstance(data, pd.Series):
        return pd.Series(mapped).reindex(data.index)
    elif isinstance(data, list):
        return [mapped[i] for i in range(len(data))]
    else:
        # in Python, hash(<int>) := <int>, so guaranteed to be in order for list
        # and tuple. For other types
        return type(data)(mapped)
