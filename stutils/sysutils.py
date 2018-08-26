
"""
Operations with files and Py2/3 compatibility tweaks
"""
import six

import os
import subprocess
from typing import Any, Optional, Tuple, Union


def mkdir(*args):
    """Create a directory specified by a sequence of subdirectories

    >>> mkdir("/tmp", "foo", "bar", "baz")
    '/tmp/foo/bar/baz'
    >>> os.path.isdir('/tmp/foo/bar/baz')
    True
    """
    path = ''
    for chunk in args:
        path = os.path.join(path, chunk)
        if not os.path.isdir(path):
            os.mkdir(path)
    return path


def shell(cmd, *args, **kwargs):
    # type: (Union[str, unicode], *Union[str, unicode], **Any) ->Tuple[int, str]
    """ Execute shell command and return output

    :param cmd: the command itself, i.e. part until the first space
    :param args: positional arguments, i.e. other space-separated parts
    :param kwargs:
            rel_path: execute relative to the path
            raise_on_status: bool, raise exception if command
                exited with non-zero status (default: `True`)
            stderr: file-like object to collect stderr output, None by default
    :return: (int status, str shell output)
    """
    if kwargs.get('rel_path') and not cmd.startswith("/"):
        cmd = os.path.join(kwargs['rel_path'], cmd)
    status = 0
    try:
        output = subprocess.check_output(
            (cmd,) + args, stderr=kwargs.get('stderr'))
    except subprocess.CalledProcessError as e:
        if kwargs.get('raise_on_status', True):
            raise e
        output = e.output
        status = e.returncode
    except OSError as e:  # command not found
        if kwargs.get('raise_on_status', True):
            raise e
        if 'stderr' in kwargs:
            kwargs['stderr'].write(e.message)
        return -1, ""

    if six.PY3:
        output = output.decode('utf8')
    return status, output


def raw_filesize(path):
    # type: (str) -> Optional[int]
    """ Get size of a file/directory in bytes.

    Will return None if path does not exist or cannot be accessed.
    """
    status, output = shell("du", "-bs", path, raise_on_status=False,
                           stderr=open('/dev/null', 'w'))
    if status != 0:
        return None
    # output is: <size>\t<path>\n
    return int(output.split("\t", 1)[0])
