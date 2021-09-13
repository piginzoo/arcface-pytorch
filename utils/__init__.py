import logging
import os
import os.path as ops
from logging import handlers

from utils.view_model import *
from utils.visualizer import *


def init_log(level=logging.DEBUG,
             when="D",
             backup=10,
             _format="%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d行 %(message)s"):
    log_path = ops.join(os.getcwd(), 'logs/arcface.log')
    _dir = os.path.dirname(log_path)
    if not os.path.isdir(_dir):
        os.makedirs(_dir)

    logger = logging.getLogger()
    if not logger.handlers:
        formatter = logging.Formatter(_format)
        logger.setLevel(level)

        handler = handlers.TimedRotatingFileHandler(log_path, when=when, backupCount=backup)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
