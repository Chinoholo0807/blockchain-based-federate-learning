import logging
import os
import sys
import time

logger = logging.getLogger('=w=')

str2level = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "FATAL": logging.FATAL,
}


def init_logging(log_dir=None, log_level_str="INFO"):
    log_level = str2level[log_level_str]
    logger.setLevel(log_level)
    log_format = '[%(levelname)-6s]%(asctime)s-%(name)-6s-%(filename)-15s-%(lineno)-4s line : %(message)s'
    date_format = '%Y-%m-%d %H-%M-%S'
    formatter = logging.Formatter(log_format, date_format)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if not (log_dir is None):
        # create file
        file_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())) + '.log'
        log_file = os.path.join(log_dir, file_name)
        with open(log_file, "w"):
            pass
        # add handler
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
