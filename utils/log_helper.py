#!/usr/bin/python

import logging

FMT = "[%(asctime)s %(levelname)s %(filename)s %(lineno)d] %(message)s"
DATEFMT = "%Y-%m-%d %H:%M:%S"

def get_logger(logger_name='model_logger', log_level='info', log_path=''):
    # create logger
    logger = logging.getLogger(logger_name)

    # create file handler
    #log_path = "./log.log"
    if not len(logger.handlers):
        if not log_path:
            handler = logging.StreamHandler()
        else:
            handler = logging.FileHandler(log_path)

        #fh.setLevel(logging.INFO)

        # create formatter
        fmt = FMT
        datefmt = DATEFMT
        formatter = logging.Formatter(fmt, datefmt)

        # add handler and formatter to logger
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if log_level == 'debug':
        logger.setLevel(logging.DEBUG)
    elif log_level == 'warn':
        logger.setLevel(logging.WARN)
    elif log_level == 'error':
        logger.setLevel(logging.ERROR)
    else:
        logger.setLevel(logging.INFO)

    return logger


