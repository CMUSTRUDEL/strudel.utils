
"""
Operations with files and Py2/3 compatibility tweaks
"""

import os
import subprocess
import sys

PY3 = sys.version_info[0] > 2

if PY3:
    from urllib.request import urlretrieve  # Python3
    import queue
    basestring = str
else:
    from urllib import urlretrieve  # Python2
    import Queue as queue
    basestring = basestring


def mkdir(*args):
    path = ''
    for chunk in args:
        path = os.path.join(path, chunk)
        if not os.path.isdir(path):
            os.mkdir(path)
    return path


def shell(cmd, *args, **kwargs):
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

    if PY3:
        output = output.decode('utf8')
    return status, output


def raw_filesize(package_dir):
    """ Get size of a file/directory in bytes
    """
    status, output = shell("du", "-bs", package_dir)
    if status != 0:
        return None
    return int(output.split(" ", 1)[0])
