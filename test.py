
from __future__ import unicode_literals, print_function

import numpy as np
import pandas as pd
import requests

import logging
import os
import random
import subprocess
import time
from typing import Generator
import unittest

from stutils import decorators as d
from stutils import email_utils as email
from stutils import mapreduce
from stutils import sysutils
from stutils import versions


def test_dataframe(x, y, *args):
    return pd.DataFrame(np.random.rand(x, y) * 10000,
                        columns=[str(i) for i in range(y)], dtype=int)


def test_series(length, *args):
    return test_dataframe(length, 1)['0']


class TestDecorators(unittest.TestCase):
    @d.cached_method
    def rand(self, length, *args):
        return pd.Series(np.random.rand(length))

    def test_cached_method(self):
        self.assertIs(self.rand(10, 'one'), self.rand(10, 'one'))
        self.assertIsNot(self.rand(10, 'one'), self.rand(10, 'two'))

    def test_memoize(self):
        mtest = d.memoize(test_series)
        self.assertIs(mtest(10, 'one'), mtest(10, 'one'))
        self.assertIsNot(mtest(10, 'one'), mtest(10, 'two'))

    def test_fs_cache(self):
        # just in case those are set in environment variables
        defaults = {
            'cache_dir': d.DEFAULT_PATH,
            'expiry': 3
        }

        # no app, no type
        cseries = d.fs_cache(**defaults)(test_series)
        self.assertIsInstance(cseries(10, 'one'), pd.Series)
        self.assertTrue((cseries(10, 'one') == cseries(10, 'one')).all())
        self.assertIsNot(cseries(10, 'one'), cseries(10, 'one'))

        cdf = d.fs_cache(**defaults)(test_dataframe)
        res = cdf(10, 2, 'one')
        self.assertIsInstance(res, pd.DataFrame)
        self.assertTrue((res == cdf(10, 2, 'one')).all(None))
        self.assertIsNot(res, cdf(10, 2, 'one'))

        # cache expiration
        time.sleep(defaults['expiry'])
        self.assertFalse((res == cdf(10, 2, 'one')).all(None))

        # app
        old_cdf = cdf
        cdf = d.fs_cache('my_app', **defaults)(test_dataframe)
        self.assertIsInstance(cdf(10, 2, 'one'), pd.DataFrame)
        self.assertTrue((cdf(10, 2, 'one') == cdf(10, 2, 'one')).all(None))
        self.assertFalse((old_cdf(10, 2, 'one') == cdf(10, 2, 'one')).all(None))
        self.assertIsNot(cdf(10, 2, 'one'), cdf(10, 2, 'one'))

        # type
        old2 = cdf
        cdf = d.fs_cache(
            'my_app', cache_type='ttype', **defaults)(test_dataframe)
        self.assertIsInstance(cdf(10, 2, 'one'), pd.DataFrame)
        self.assertTrue((cdf(10, 2, 'one') == cdf(10, 2, 'one')).all(None))
        self.assertFalse((old_cdf(10, 2, 'one') == cdf(10, 2, 'one')).all(None))
        self.assertFalse((old2(10, 2, 'one') == cdf(10, 2, 'one')).all(None))
        self.assertIsNot(cdf(10, 2, 'one'), cdf(10, 2, 'one'))

        # multilevel index
        def test_multiindex_df(*args):
            return test_dataframe(10, 4, *args).set_index(['0', '1'])

        def test_multiindex_series(*args):
            return test_multiindex_df(*args)['2']

        cseries = d.fs_cache(idx=2, **defaults)(test_multiindex_series)
        self.assertIsInstance(cseries(10, 'one'), pd.Series)
        self.assertIsInstance(cseries(10, 'one').index, pd.MultiIndex)
        self.assertTrue((cseries(10, 'one') == cseries(10, 'one')).all())
        self.assertIsNot(cseries(10, 'one'), cseries(10, 'one'))

        cdf = d.fs_cache(idx=2, **defaults)(test_multiindex_df)
        self.assertIsInstance(cdf(10, 2, 'one'), pd.DataFrame)
        self.assertIsInstance(cdf(10, 2, 'one').index, pd.MultiIndex)
        self.assertTrue((cdf(10, 2, 'one') == cdf(10, 2, 'one')).all(None))
        self.assertIsNot(cdf(10, 2, 'one'), cdf(10, 2, 'one'))

        cdf = d.fs_cache(idx=['0', '1'], **defaults)(test_multiindex_df)
        self.assertIsInstance(cdf(10, 2, 'two'), pd.DataFrame)
        self.assertIsInstance(cdf(10, 2, 'two').index, pd.MultiIndex)
        self.assertTrue((cdf(10, 2, 'two') == cdf(10, 2, 'two')).all(None))
        self.assertIsNot(cdf(10, 2, 'two'), cdf(10, 2, 'two'))

    def test_cache_iterator(self):
        # just in case those are set in environment variables
        defaults = {
            'cache_dir': d.DEFAULT_PATH,
            'expiry': 3
        }

        def test_iterator(*args):
            if args and args[0] == 'empty':
                return
            yield "string"
            yield random.randint(0, 10000)
            yield ['one', 'tow', 1]

        # no app, no type
        iter = d.cache_iterator(**defaults)(test_iterator)
        res = iter('one')
        res_list = list(res)
        res2 = iter('one')
        res2_list = list(res2)
        res3_list = list(iter('two'))
        # check result is still a generator
        self.assertIsInstance(res, Generator)
        self.assertIsInstance(res2, Generator)
        self.assertTrue(res_list == res2_list)
        self.assertIsNot(res, res2)
        self.assertFalse(res_list == res3_list)

        res4 = iter('empty')
        res4_list = list(res4)
        res5_list = list(iter('empty'))
        self.assertIsInstance(res4, Generator)
        self.assertTrue(res4_list == res5_list)

    def test_threadpool(self):
        n = 1000

        @d.threadpool(4)  # sleep in four threads
        def increment(*args):
            time.sleep(0.005)

        start_time = time.time()
        increment(list(range(n)))
        elapsed = time.time() - start_time
        self.assertLess(elapsed, 0.005 * n * 0.3)

    def test_guard(self):
        n = 1000
        data = {'counter': 0}

        @d.threadpool(4)
        @d.guard
        def increment(*args):
            i = data['counter']
            time.sleep(0.005)
            data['counter'] = i + 1

        increment(list(range(n)))
        self.assertEqual(data['counter'], n)


class TestMapReduce(unittest.TestCase):

    def setUp(self):
        users = ("pandas-dev", "numpy", "django", "requests", "saltstack",
                 "keras-team", "ansible", "scikit-learn", "conda", "scipy",
                 "gevent", "tornadoweb", "aio-libs", "lxml", "python-pillow",
                 "chardet", "pallets", "zopefoundation", "openstack",
                 "alskdfjalskd", "owietmqwoinvq")
        self.urls = ["https://github.com/orgs/%s/people" % user
                     for user in users]

        start = time.time()
        self.reference_results = [requests.get(url).status_code
                                  for url in self.urls]
        self.baseline_perf = time.time() - start
        # logging.basicConfig(level=logging.INFO)

    def test_thread_pool(self):
        tp = mapreduce.ThreadPool(8)
        results_dict = {}

        def callback(res):
            status, url = res
            results_dict[url] = status

        def do(url):
            return requests.get(url).status_code, url

        start = time.time()
        _ = [tp.submit(do, url, callback=callback) for url in self.urls]
        tp.shutdown()
        results = [results_dict[url] for url in self.urls]
        perf = time.time() - start

        self.assertSequenceEqual(self.reference_results, results)
        self.assertGreater(self.baseline_perf/perf, 3.5)
        logging.info("Async mapping: %s seconds, sync baseline: %s"
                     "" % (perf, self.baseline_perf))

    def test_native_threadpool(self):
        # this test shows native Threadpool is 30..50% slower
        from multiprocessing.pool import ThreadPool

        def do(url):
            return requests.get(url).status_code

        pool = ThreadPool()
        start = time.time()
        results = list(pool.imap(do, self.urls))
        perf = time.time() - start
        self.assertSequenceEqual(self.reference_results, results)
        logging.info("Native ThreadPool mapping: %s seconds, sync baseline: %s"
                     "" % (perf, self.baseline_perf))


''


class TestEmailUtils(unittest.TestCase):

    def test_parse(self):
        self.assertEqual(email.parse(
            '<me@someorg.com@ce2b1a6d-e550-0410-aec6-3dcde31c8c00>'),
            ('me', 'someorg.com'))
        self.assertRaises(email.InvalidEmail, lambda: email.parse(42))
        self.assertRaises(email.InvalidEmail, lambda: email.parse("boo"))

    def test_clean(self):
        self.assertEqual(email.clean('me@someorg.com'), 'me@someorg.com')
        self.assertEqual(email.clean('<me@someorg.com'), 'me@someorg.com')
        self.assertEqual(email.clean('me@someorg.com>'), 'me@someorg.com')
        self.assertEqual(
            email.clean('John Doe <me@someorg.com>'), 'me@someorg.com')
        self.assertEqual(
            email.clean('John Doe me@someorg.com'), 'me@someorg.com')
        # git2svn produces addresses like this:
        self.assertEqual(email.clean(
            '<me@someorg.com@ce2b1a6d-e550-0410-aec6-3dcde31c8c00>'),
            'me@someorg.com')
        self.assertEqual(email.clean("invalid email"), None)

    def test_domain(self):
        self.assertEqual(email.domain(
            'Missing test@dep.uni.edu@ce2b1a6d-e550-0410-aec6-3dcde31c8c00>'),
            'dep.uni.edu')
        self.assertEqual(email.domain("invalid email"), None)

    def test_university_domains(self):
        self.assertIsInstance(email.university_domains(), set)
        # 4902 as of Jan 2018
        self.assertGreater(len(email.university_domains()), 4000)
        # .edu domains are excluded
        self.assertNotIn('cmu.edu', email.university_domains())
        self.assertIn('upsa.es', email.university_domains())

    def test_public_domains(self):
        self.assertIsInstance(email.public_domains(), set)
        # 3628 as of Jan 2018
        self.assertGreater(len(email.public_domains()), 3000)
        self.assertFalse(
            email.public_domains().intersection(email.university_domains()))
        self.assertNotIn('google.com', email.public_domains())
        self.assertIn('yandex.ru', email.public_domains())
        self.assertIn('163.com', email.public_domains())

    def test_domain_user_stats(self):
        self.assertIsInstance(email.domain_user_stats(), pd.Series)
        self.assertGreater(len(email.domain_user_stats()), 100000)
        self.assertTrue((email.domain_user_stats() > 0).all())

    def test_commercial_domains(self):
        self.assertIn("google.com", email.commercial_domains())  # commercial
        self.assertNotIn("gmail.com", email.commercial_domains())  # public
        self.assertNotIn("jaraco.com", email.commercial_domains())  # personal
        self.assertNotIn("cmu.edu", email.commercial_domains())  # university

    def test_domain_intersections(self):
        self.assertFalse(email.public_domains().intersection(
            email.commercial_domains()))
        self.assertFalse(email.public_domains().intersection(
            email.university_domains()))
        self.assertFalse(email.commercial_domains().intersection(
            email.university_domains()))

    def test_is_university(self):
        self.assertTrue(email.is_university("john@abc.cmu.edu"))
        self.assertTrue(email.is_university("john@abc.edu.uk"))
        self.assertTrue(email.is_university("john@edu.au"))
        self.assertFalse(email.is_university("john@aedu.au"))
        self.assertTrue(email.is_university("john@vvsu.ru"))
        self.assertTrue(email.is_university("john@abc.vvsu.ru"))
        self.assertFalse(email.is_university("john@england.edu"))
        self.assertFalse(email.is_university("John Doe"))
        #  two cool exceptions
        self.assertFalse(email.is_university("john@australia.edu"))
        self.assertFalse(email.is_university("john@england.edu"))

    def test_is_public(self):
        self.assertTrue(email.is_public("john@163.com"))
        self.assertTrue(email.is_public("john@qq.com"))
        self.assertFalse(email.is_public("john@abc.vvsu.ru"))
        self.assertTrue(email.is_public("john@localhost"))
        self.assertTrue(email.is_public("John Doe"))
        # two cool exceptions
        self.assertTrue(email.is_public("john@australia.edu"))
        self.assertTrue(email.is_public("john@england.edu"))
        # new TLDs might be really long
        self.assertFalse(email.is_public("john@ozon.travel"))

    def test_is_commercial(self):
        self.assertFalse(email.is_commercial("john@163.com"))  # public
        self.assertFalse(email.is_commercial("john@localhost"))  # local
        self.assertFalse(email.is_commercial("john@abc.vvsu.ru"))  # academic
        self.assertFalse(email.is_commercial("john@jaraco.com"))  # personal
        self.assertFalse(email.is_commercial("John Doe"))
        self.assertTrue(email.is_commercial("john@hp.com"))
        self.assertTrue(email.is_commercial("john@google.com"))
        self.assertTrue(email.is_commercial("john@microsoft.com"))
        #  two cool exceptions
        self.assertFalse(email.is_commercial("john@australia.edu"))
        self.assertFalse(email.is_commercial("john@england.edu"))

    def test_bulk_methods(self):
        s = email.domain_user_stats().iloc[:300]
        es = pd.Series("test@" + s.index, index=s.index)
        is_public = email.is_public_bulk(es)
        self.assertIsInstance(is_public, pd.Series)
        self.assertEqual(len(is_public), len(es))
        is_university = email.is_university_bulk(es)
        self.assertIsInstance(is_university, pd.Series)
        self.assertEqual(len(is_university), len(es))
        is_commercial = email.is_commercial_bulk(es)
        self.assertIsInstance(is_commercial, pd.Series)
        self.assertEqual(len(is_commercial), len(es))


class TestSysutils(unittest.TestCase):

    def test_shell(self):
        # system command
        status, output = sysutils.shell("echo", "Hello world!")
        self.assertEqual(status, 0)
        self.assertEqual(output.strip(), "Hello world!")

        # relative file path
        rel_path = os.path.dirname(__file__) or '.'
        status, output = sysutils.shell("LICENSE", rel_path=rel_path)
        self.assertEqual(status, 0)
        self.assertGreater(len(output), 35000)

        # invalid commands
        stderr = open('/dev/null', 'wb')
        with self.assertRaises(OSError):
            sysutils.shell("ls", rel_path=".", stderr=stderr)

        with self.assertRaises(subprocess.CalledProcessError):
            sysutils.shell("ls", "eatmyshorts", stderr=stderr)

        status, output = sysutils.shell("eatmyshorts", raise_on_status=False)
        self.assertEqual(status, -1)

        status, output = sysutils.shell(
            "ls", "eatmyshorts", raise_on_status=False, stderr=stderr)
        self.assertEqual(status, 2)
        stderr.close()

    def test_filesize(self):
        rel_path = os.path.dirname(__file__) or '.'

        self.assertGreater(sysutils.raw_filesize(rel_path), 100000)
        self.assertIsNone(sysutils.raw_filesize(rel_path + '/eatmyshorts'))


class TestVersions(unittest.TestCase):

    def test_is_alpha(self):
        self.assertFalse(versions.is_alpha("1.0.0"))
        self.assertTrue(versions.is_alpha("1.0rc1"))

    def test_parse(self):
        self.assertEqual(versions.parse("1"), [1])
        self.assertEqual(versions.parse("0.0.1"), [0, 0, 1])
        self.assertEqual(versions.parse("0.11.23.rc1"), [0, 11, 23, 'rc1'])

    def test_compare(self):
        self.assertEqual(versions.compare("0.1.1", "0.1.2"), -1)
        self.assertEqual(versions.compare("0.1.2", "0.1.1"), 1)
        self.assertEqual(versions.compare("0.1", "0.1.1"), 0)
        self.assertEqual(versions.compare("0.1.1rc1", "0.1.1a"), 1)
        self.assertEqual(versions.compare("0.1.1rc1", "0.1.1"), -1)


if __name__ == "__main__":
    unittest.main()
