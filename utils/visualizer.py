import logging

import numpy as np
import visdom
from torch.utils.tensorboard import SummaryWriter
import os
import utils

logger = logging.getLogger(__name__)


class Visualizer(object):
    """
    使用visdom可视化，visdom是PyTorch的远程可视化神器：https://zhuanlan.zhihu.com/p/34692106
    我自己的评价，visdom不好用,原因：
    1、不支持提前写入，然后像tensorboard那样，后起一个服务器来读取
    2、创建Visdom实例的时候，自动连接，而且还报错，无法catch住，很恼人，逼的我都先做个端口check才可以
    3、图形方面，没体验，待评价...
    """

    def __init__(self, config):
        if config.visualizer == "tensorboard":
            self.visualizer = TensorboardVisualizer(config.tensorboard_dir)
            return
        if config.visualizer == "visdom":
            self.visualizer = VisdomVisualizer('arcface', config.visdom_port)
            return
        raise ValueError("无法识别的Visualizer类型：" + config.visualizer)

    def write(self, step, value, name):
        self.visualizer.write(step, value, name)


class TensorboardVisualizer(object):
    """
    参考：https://jishuin.proginn.com/p/763bfbd5447c
    """

    def __init__(self, log_dir):
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        self.summaryWriter = SummaryWriter(log_dir=log_dir)

    def write(self, step, value, name):
        self.summaryWriter.add_scalar(tag=name, scalar_value=value, global_step=step)


class VisdomVisualizer(object):

    def __init__(self, env, port):
        if not utils.check_port_in_use(port):
            logger.error("创建Visualizer连接失败，端口无法连接：%r", port)
            self.vis = None
            return

        self.vis = visdom.Visdom(env=env, port=port)
        if not self.vis.check_connection():
            logger.error("无法连接到visdom服务器")
        self.vis.close()

        self.iters = {}
        self.lines = {}
        self.env = env

    def check_connection(self):
        if not self.vis.check_connection():
            logger.warning("无法连接到visdom服务器,无法写入")
            return False
        return True

    def write(self, iters, x, name='train_loss'):
        if not self.check_connection(): return

        if name not in self.iters:
            self.iters[name] = []

        if name not in self.lines:
            self.lines[name] = []

        self.iters[name].append(iters)
        self.lines[name].append(x)

        self.vis.line(X=np.array(self.iters[name]),
                      Y=np.array(self.lines[name]),
                      win=name,
                      opts=dict(legend=[name], title=name))
        self.vis.save([self.env])
