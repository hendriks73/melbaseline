from os import makedirs, remove
from os.path import dirname, isdir
from random import randint
import re

from tensorflow.python.lib.io.file_io import file_exists, FileIO


def create_local_copy(remote_file):
    """
    Create a local copy for the given, potentially remote file.
    Does nothing, if the file is already local.

    :param remote_file: potentially remote file
    :return: local file
    """
    local_file = remote_file
    if not file_exists(remote_file):
        print('File does not exist: {}'.format(remote_file))

    if is_remote(remote_file):
        local_file = to_local(remote_file)
        print('Writing to local file: {}'.format(local_file))
        copy(remote_file, local_file)
    else:
        print('Local copy for {} not needed.'.format(remote_file))
    return local_file


def copy(source, dest):
    """
    Copy from source to dest, create all necessary dirs.

    :param source: source file
    :param dest: dest file
    """
    with FileIO(source, mode='rb') as input_f:
        if '/' in dest and not isdir(dirname(dest)):
            makedirs(dirname(dest))
        with open(dest, mode='wb') as output_f:
            while 1:
                buf = input_f.read(1024 * 1024)
                if not buf:
                    break
                output_f.write(buf)


def is_remote(file):
    return file is not None and file.startswith('gs://')


def to_local(file):
    """
    Remove the ``gs://BUCKET/`` prefix.

    :param file: file
    :return: relative file name
    """
    pattern = re.compile('gs://[^/]+/(.*)')
    match = pattern.match(file)
    return match.group(1)


def save_model(model, file):
    """
    Save model to the given file (potentially Google storage).

    :param model: model
    :param file: output file
    """
    print('Saving model to file {}.'.format(file))
    temp_file = 'temp_model_{}.h5'.format(randint(0, 100000000))
    model.save(temp_file)
    try:
        # copy model to google storage
        with FileIO(temp_file, mode='rb') as input_f:
            with FileIO(file, mode='wb') as output_f:
                output_f.write(input_f.read())
    finally:
        remove(temp_file)
