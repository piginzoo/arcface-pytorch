# 参考 https://github.com/amirhfarzaneh/lsoftmax-pytorch/blob/master/train_mnist.py
import logging
import warnings

import cv2
import matplotlib
import numpy as np

import utils

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').disabled = True
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="matplotlib")
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

logger = logging.getLogger(__name__)

# python -m sandbox.test_tboard_image
if __name__ == '__main__':
    utils.init_log()
    visualizer = utils.TensorboardVisualizer("logs/tboard")

    data = np.random.randint(0, 10, (100, 2))
    labels = np.random.randint(0, 10, (100))
    visualizer.text(0, 1.0, name='train_loss')
    visualizer.text(0, 2.0, name='train_acc')
    image = cv2.imread("data/train/Img/img_align_celeba/010353.jpg")
    logger.debug(image.shape)
    image = np.transpose(image, (2, 0, 1))
    logger.debug(image.shape)
    images = np.array([image])
    visualizer.image(images, name='train_images')
    visualizer.plot_2d_embedding("test_plot", data, labels, 0)
