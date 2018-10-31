
import ijson.backends.yajl2 as ijson
import pandas as pd

import json
import logging
import os
import time
from functools import wraps
import tempfile

import stutils
from stutils.sysutils import mkdir

DEFAULT_EXPIRY = stutils.get_config('ST_FS_CACHE_DURATION', 3600 * 24 * 30 * 3)
DEFAULT_PATH = os.path.join(tempfile.gettempdir(), '.st_fs_cache')


def _argstring(*args):
    """Convert a list of variables into a single string for naming cache files.
    It is used internally by many caching decorators
    """
    return "_".join([str(arg).replace("/", ".") for arg in args])


class fs_cache(object):
    """ A decorator to cache results of functions returning
    pd.DataFrame or pd.Series objects

    Assuming you've configured `ST_FS_CACHE_PATH`,

    Basic use case: just use file caching
    (`get_data` cache files will be stored directly at `${ST_FS_CACHE_PATH}/`)
    >>> @fs_cache()
    ... def get_data(*args):
    ...     return pd.DataFrame(range(10))

    Separate cache by applications:
    (`get_data` cache files will be stored at `${ST_FS_CACHE_PATH}/my_app`)
    >>> myapp_fs_cache = fs_cache('my_app')
    >>> @myapp_fs_cache
    ... def get_data(*args):
    ...     return pd.DataFrame(range(10))

    Separate cache by applications and type:
    (`get_data` cache files will be stored at
        `${ST_FS_CACHE_PATH}/my_app/view_stats/`)
    >>> myapp_fs_cache = fs_cache('my_app', cache_type='view_stats')
    >>> @myapp_fs_cache
    ... def get_data(*args):
    ...     return pd.DataFrame(range(10))

    Multiindex support: specify number of levels with `idx`:
    >>> import numpy as np
    >>> myapp_fs_cache = fs_cache(idx=2)
    >>> @myapp_fs_cache
    ... def get_data(*args):
    ...     return pd.DataFrame(np.random.rand(10, 4)).set_index([0, 1])
    """

    def __init__(self, app_name=None, idx=1, cache_type='',
                 expires=DEFAULT_EXPIRY, ds_path=None):
        """

        :param app_name: if present, cache files for this application will be
            stored in a separate folder
        :param idx: number of columns to use as an index
        :param cache_type: if present, cache files within app directory will be
            separated into different folders by their cache_type.
        :param expires: cache duration in seconds
        :param ds_path: set custom file cache path

        I.e., by default all cache files will be stored in the path specified by
        `ST_FS_CACHE_PATH`. If app_name is specified for the decorator
        (`@fs_cache('my_app')`), cache files for the decorated function will be
        stored in `${ST_FS_CACHE_PATH}/my_app`. If

        """
        if not ds_path:
            ds_path = stutils.get_config('ST_FS_CACHE_PATH', DEFAULT_PATH)

        self.expires = expires
        self.idx = idx
        if not app_name:
            self.cache_path = ds_path
        else:
            self.cache_path = mkdir(ds_path, app_name + ".cache", cache_type)
        if not os.path.isdir(self.cache_path):
            mkdir(self.cache_path)

    def get_cache_fname(self, func_name, *args, **kwargs):
        chunks = [func_name]
        if args:
            chunks.append(_argstring(*args))
        chunks.append(kwargs.get("extension", "csv"))
        return os.path.join(self.cache_path, ".".join(chunks))

    def expired(self, cache_fpath):
        return not os.path.isfile(cache_fpath) \
               or time.time() - os.path.getmtime(cache_fpath) > self.expires

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args):
            cache_fpath = self.get_cache_fname(func.__name__, *args)

            if not self.expired(cache_fpath):
                return pd.read_csv(cache_fpath, index_col=range(self.idx),
                                   encoding="utf8", squeeze=True)

            res = func(*args)
            if isinstance(res, pd.DataFrame):
                df = res
                if len(df.columns) == 1 and self.idx == 1:
                    logging.warning(
                        "Single column dataframe is returned by %s.\nSince it "
                        "will cause inconsistent behavior with @fs_cache "
                        "decorator, please consider changing result type "
                        "to pd.Series", func.__name__)
                if any(not isinstance(cname, str) for cname in df.columns):
                    logging.warning(
                        "Some of the dataframe columns aren't strings. "
                        "This will result in inconsistent naming if read from "
                        "filesystem cache.")
            elif isinstance(res, pd.Series):
                df = pd.DataFrame(res)
            else:
                raise ValueError("Unsupported result type (pd.DataFrame or "
                                 "pd.Series expected, got %s)" % type(res))
            df.to_csv(cache_fpath, float_format="%g", encoding="utf-8")
            return res
        return wrapper

    def invalidate(self, func):
        """ Remove all files caching this function """
        for fname in os.listdir(self.cache_path):
            if fname.startswith(func.__name__):
                os.remove(os.path.join(self.cache_path, fname))


def typed_fs_cache(app_name, expires=DEFAULT_EXPIRY):
    # type: (str, int) -> callable
    """A convenience wrapper over fs_cache to separate cache files from
    different applications into different folders.
    """
    def _cache(cache_type, idx=1):
        return fs_cache(app_name, idx, cache_type=cache_type, expires=expires)

    return _cache


def memoize(func):
    """ Classic memoize decorator for non-class methods """
    cache = {}

    @wraps(func)
    def wrapper(*args):
        key = "__".join(str(arg) for arg in args)
        if key not in cache:
            cache[key] = func(*args)
        return cache[key]
    return wrapper


def cached_method(func):
    """ Memoize for class methods """
    @wraps(func)
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


class cache_iterator(fs_cache):
    """ A modification of fs_cache to handle large unstructured iterators
        - e.g., a result of a GitHubAPI call
    """
    def __call__(self, func):
        @wraps(func)
        def wrapper(*args):
            cache_fpath = self.get_cache_fname(
                func.__name__, *args, **{'extension': 'json'})

            if not self.expired(cache_fpath):
                cache_fh = open(cache_fpath, 'rb')
                for item in ijson.items(cache_fh, "item"):
                    yield item
            else:
                # if iterator is not exhausted, the resulting file
                # will contain invalid JSON. So, we write to a tempfile
                # and rename when the iterator is exhausted
                cache_fh = tempfile.NamedTemporaryFile(delete=False)
                sep = "[\n"
                for item in func(*args):
                    cache_fh.write(sep)
                    sep = ",\n"
                    cache_fh.write(json.dumps(item))
                    yield item
                cache_fh.write("]")
                cache_fh.close()
                os.rename(cache_fh.name, cache_fpath)

        return wrapper
