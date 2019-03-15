
import json
import logging
import os
import shutil
import time
import functools
import tempfile
import threading

import pandas as pd
import stutils
from stutils import mapreduce
from stutils.sysutils import mkdir

# filesystem cache decorator defaults
DEFAULT_EXPIRY = stutils.get_config('ST_FS_CACHE_DURATION', 3600 * 24 * 30 * 3)
DEFAULT_PATH = stutils.get_config(
    'ST_FS_CACHE_PATH', os.path.join(tempfile.gettempdir(), '.st_fs_cache'))


def _argstring(*args):
    """Convert a list of variables into a single string for naming cache files.
    It is used internally by many caching decorators
    """
    return "_".join([str(arg).replace("/", ".") for arg in args])


class _FSCacher(object):
    extension = 'csv'

    def __init__(self, func, cache_dir='', app_name='', cache_type='', idx=1,
                 expiry=None):
        # type: (callable, str, str, str, Union[int, str, list], int) -> None
        """ Helper class for @fs_cache internals
        """
        self.func = func
        functools.update_wrapper(self, func)
        self.expiry = expiry or DEFAULT_EXPIRY
        # Will create a path:
        # <cache_dir>/<app_name>/<cache_type>/, omitting missing parts
        self.cache_path = mkdir(cache_dir or DEFAULT_PATH, app_name, cache_type)

        if isinstance(idx, int):
            idx = range(idx)
        self.idx = idx

    def get_cache_fname(self, *args):
        chunks = [self.func.__name__]
        if args:
            chunks.append(_argstring(*args))
        chunks.append(self.extension)
        return os.path.join(self.cache_path, ".".join(chunks))

    def _expired(self, cache_fpath):
        return not os.path.isfile(cache_fpath) \
               or time.time() - os.path.getmtime(cache_fpath) > self.expiry

    def __call__(self, *args):
        cache_fpath = self.get_cache_fname(*args)

        if not self._expired(cache_fpath):
            return pd.read_csv(cache_fpath, index_col=self.idx,
                               encoding="utf8", squeeze=True)

        res = self.func(*args)
        if isinstance(res, pd.DataFrame):
            df = res
            if len(df.columns) == 1 and self.idx == 1:
                logging.warning(
                    "Single column dataframe is returned by %s.\nSince "
                    "it will cause inconsistent behavior with @fs_cache "
                    "decorator, please consider changing result type "
                    "to pd.Series", self.func.__name__)
            if any(not isinstance(cname, str) for cname in df.columns):
                logging.warning(
                    "Some of the dataframe columns aren't strings. "
                    "This will result in inconsistent naming "
                    "if read from filesystem cache.")
        elif isinstance(res, pd.Series):
            # starting pandas 0.25 DataFrame and Series to_csv()
            # will become compatible, but at least until 0.30 comes out
            # keep this for compatibility
            df = pd.DataFrame(res)
        else:
            raise ValueError("Unsupported result type (pd.DataFrame or "
                             "pd.Series expected, got %s)" % type(res))
        df.to_csv(cache_fpath, float_format="%g", encoding="utf-8")
        return res

    def expired(self, *args):
        return self._expired(self.get_cache_fname(*args))

    def cached(self, *args):
        return not self.expired(*args)

    def invalidate(self, *args):
        try:
            os.remove(self.get_cache_fname(*args))
        except OSError:
            return False
        return True

    def invalidate_all(self):
        """ Remove all files caching this function """
        for fname in os.listdir(self.cache_path):
            if fname.startswith(self.func.__name__ + "."):
                os.remove(os.path.join(self.cache_path, fname))


class fs_cache(object):
    """
    A decorator to cache results of functions returning
    pd.DataFrame or pd.Series objects under:
    <cache_dir>/<app_name>/<cache_type>/<func_name>.<param_string>.csv,
    missing parts, like app_name and cache_type, will be omitted

    If cache_dir is omitted, stutils 'ST_FS_CACHE_PATH' conf dir will be used.
    If 'ST_FS_CACHE_PATH' is not configured, a temporary directory
    will be created.

    :param app_name: if present, cache files for this application will be
        stored in a separate folder
    :param idx: number of columns to use as an index
    :param cache_type: if present, cache files within app directory will be
        separated into different folders by their cache_type.
    :param expiry: cache duration in seconds
    :param cache_dir: set custom file cache path
    """
    helper_class = _FSCacher

    def __init__(self, app_name='', idx=1, cache_type='', expiry=DEFAULT_EXPIRY,
                 cache_dir=''):
        self.app_name = app_name
        self.idx = idx
        self.cache_type = cache_type
        self.expiry = expiry
        self.cache_dir = cache_dir

    def __call__(self, func):
        return self.helper_class(func, self.cache_dir, self.app_name,
                                 self.cache_type, self.idx, self.expiry)


def memoize(func):
    """ Classic memoize decorator for non-class methods """
    cache = {}

    @functools.wraps(func)
    def wrapper(*args):
        key = "__".join(str(arg) for arg in args)
        if key not in cache:
            cache[key] = func(*args)
        return cache[key]
    return wrapper


def cached_method(func):
    """ Memoize for class methods """
    @functools.wraps(func)
    def wrapper(self, *args):
        if not hasattr(self, "_cache"):
            self._cache = {}
        key = _argstring((func.__name__,) + args)
        if key not in self._cache:
            self._cache[key] = func(self, *args)
        return self._cache[key]
    return wrapper


def cached_property(func):
    return property(cached_method(func))


class _FSIterCacher(_FSCacher):
    extension = 'json'

    def __init__(self, *args, **kwargs):
        try:
            import ijson.backends.yajl2 as ijson
        except ImportError:
            raise ImportError("Please install yajl-tools to use this decorator")
        self.ijson = ijson
        super(_FSIterCacher, self).__init__(*args, **kwargs)

    def __call__(self, *args):
        cache_fpath = self.get_cache_fname(*args)

        if not self._expired(cache_fpath):
            with open(cache_fpath, 'rb') as cache_fh:
                for item in self.ijson.items(cache_fh, "item"):
                    yield item
            return

        # if the iterator is not exhausted, the resulting file
        # will contain invalid JSON. So, we write to a tempfile
        # and then rename after the iterator is exhausted
        cache_fh = tempfile.TemporaryFile()
        cache_fh.write("[\n".encode('utf8'))
        sep = ""
        for item in self.func(*args):
            cache_fh.write(sep.encode('utf8'))
            sep = ",\n"
            cache_fh.write(json.dumps(item).encode('utf8'))
            yield item
        cache_fh.write("]".encode('utf8'))
        cache_fh.flush()
        # os.rename will fail if /tmp is mapped to a different device
        cache_fh.seek(0, 0)
        target_fh = open(cache_fpath, 'wb')
        shutil.copyfileobj(cache_fh, target_fh)
        target_fh.close()
        cache_fh.close()


class cache_iterator(fs_cache):
    """ A modification of fs_cache to handle large unstructured iterators
        - e.g., a result of a GitHubAPI call

    Special cases:
        json always saves dict keys as strings, so cached dictionaries aren't
            exactly the same as original
        in Python2, json instantiates loaded strings as unicode, so cached
            result might be slightly different from original
    """
    helper_class = _FSIterCacher


def guard(func):
    """ Prevents the decorated function from parallel execution.

     Internally, this decorator creates a Lock object and transparently
     obtains/releases it when calling the function.
     """
    semaphore = threading.Lock()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        semaphore.acquire()
        try:
            return func(*args, **kwargs)
        finally:
            semaphore.release()

    return wrapper


def threadpool(num_workers=None):
    """Apply stutils.mapreduce.map to the given function"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(data):
            return mapreduce.map(func, data, num_workers)
        return wrapper
    return decorator
