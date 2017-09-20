import datetime
import hashlib
import logging
import sys


VERBOSIT = logging.DEBUG


def get_logger(name):# Configure log sytle
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(VERBOSIT)
    ch.setFormatter(logging.Formatter('[%(asctime)s][%(name)s-%(levelname)s] %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(VERBOSIT)
    logger.addHandler(ch)
    return logger


def calculate_sha256(path):
    with open(path, 'rb') as f:
        sh = hashlib.sha256()
        sh.update(f.read())
        return sh.hexdigest()


def get_timestr():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
