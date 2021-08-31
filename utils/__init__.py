import logging

from utils.visualizer import *
from utils.view_model import *

def init_log():
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                        level=logging.DEBUG,
                        handlers=[logging.StreamHandler()])
