import logging
import re
from pathlib import Path
import os

LOGGING_FIELD_SEPARATOR = ':'
LOGGING_ROOT=os.environ["HOME"] + '/logs/' # convention is to use / to follow directory
file_formatter = logging.Formatter('%(asctime)s: {%(pathname)s:%(lineno)d} :%(message)s')
Path(LOGGING_ROOT).mkdir(exist_ok=True)
file_handler = logging.FileHandler(LOGGING_ROOT + 'default.log')
file_handler.setLevel(logging.ERROR)
file_handler.setFormatter(file_formatter)

stream_formatter = logging.Formatter('{%(pathname)s:%(lineno)d} :%(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(stream_formatter)


def get_line_count(inFile):
    count = -1
    for count, line in enumerate(open(inFile, 'r')):
        pass
    count += 1
    return count

import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def make_parentdir(path):
    mkdir_p(os.path.dirname(path))


def getLogger(name='', file_handler=file_handler, stream_handler=stream_handler):
    # add in the name along with a LOGGING_FIELD_SEPARATOR
    # if name:
    #     if not name.endswith(LOGGING_FIELD_SEPARATOR):
    #         name += LOGGING_FIELD_SEPARATOR
    logger = logging.getLogger(name)

    if file_handler:
        logger.addHandler(file_handler)

    # initialize the logger level to be low
    if stream_handler:
        logger.addHandler(stream_handler)

    # prevent double printing
    # https://docs.python.org/2/library/logging.html#logger-objects
    logger.propagate = False

    # logger.error('starting') # starting logging
    logger.setLevel(logging.DEBUG)
    return logger


def condenseSpace(s):
    return re.sub('([\s])+', '\g<1>', s)


def flatten(listOfLists):
    """
    :param listOfLists:
    :return:
    >>> flatten([[1,2], [1, 3]])
    [1, 2, 1, 3]
    """
    return [item for sublist in listOfLists for item in sublist]

re_nonLetter = re.compile('[^a-zA-Z]')

def removeNonLetter(doc, replaceWithSpace=False):
    if replaceWithSpace:
        doc = re.sub(re_nonLetter, ' ', doc)
    else:
        doc = re.sub(re_nonLetter, '', doc)
    # doc = ''.join(i for i in text if ord(i)<128)
    return doc
