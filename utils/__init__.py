import os.path as ops
import socket
from logging import handlers

import torch

from config import Config
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


def check_port_in_use(port, host='127.0.0.1'):
    s = None
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(1)
        s.connect((host, int(port)))
        return True
    except socket.error:
        return False
    finally:
        if s:
            s.close()


def load_image(image_path):
    """
    使用cv2做处理：
    1、BGR=>RGB
    2、[H,W,3]=> [3,H,W]（注意，H在前，这和torch中要求的顺序一致：Image Tensor in the form [C, H, W].）
    3、做Normalize
    """
    if not os.path.exists(image_path):
        logger.warning("图片路径不存在：%s", image_path)
        return None

    # 加载成黑白照片
    image = cv2.imread(image_path)
    if image is None:
        logger.warning("图片加载失败：%s", image_path)
        return None

    # resize
    image = cv2.resize(image, Config().input_shape[1:])
    image = image.astype(np.float32, copy=False)
    # 归一化
    image -= 127.5
    image /= 127.5

    image = np.transpose(image, (2, 0, 1))  # [H,W,3] => [3,W,H]
    image = image[:, :, ::-1]  # BGR => RGB

    # logger.debug("加载了图片: %s => %r", image_path, image.shape)
    return image


def load_model(model_path):
    model = torch.load(model_path)
    logger.info("加载模型：%s", model_path)
    return model
