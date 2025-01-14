import io
import logging
import os
import warnings
from datetime import datetime

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import visdom
from tensorboard.plugins import projector

import utils

logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib').disabled = True
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="matplotlib")
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示为方块的问题

logger = logging.getLogger(__name__)


class TensorboardVisualizer(object):
    """
    参考：https://jishuin.proginn.com/p/763bfbd5447c
    """

    def __init__(self, log_dir):
        self.log_dir = os.path.join(log_dir, datetime.strftime(datetime.now(), '%Y%m%d-%H%M'))

        if not os.path.exists(self.log_dir):    os.makedirs(self.log_dir)

    def text(self, step, value, name):
        summary_writer = tf.summary.create_file_writer(logdir=self.log_dir)
        with summary_writer.as_default():
            tf.summary.scalar(name, value, step=step)
        summary_writer.close()

    def image(self, images, name, step=0):
        """
        :param images: `[b, h, w, c]`
        """
        if type(images) != np.ndarray:
            raise ValueError("图像必须为numpy数组，当前图像为：" + str(type(images)))
        if len(images.shape) == 3:
            images = images[np.newaxis, :]
        if len(images.shape) != 4:
            raise ValueError("图像必须为[B,H,W,C]，当前图像为：" + str(images.shape))

        summary_writer = tf.summary.create_file_writer(logdir=self.log_dir)
        with summary_writer.as_default():
            if images.shape[1] == 3:  # [B,C,H,W]
                images = np.transpose(images, (0, 2, 3, 1))  # [B,C,H,W]=>[B,H,W,C], tf2.x的image通道顺序
            r = tf.summary.image(name, images, step)
        summary_writer.close()
        if not r: logger.error("保存图片到tensorboard失败：%r", images.shape)
        return r

    # 参考 https://github.com/amirhfarzaneh/lsoftmax-pytorch/blob/master/train_mnist.py
    def plot_2d_embedding(self, name, features, labels, step):
        """
        feature : numpy array, shape [N, 2],是一个被降维为2的一个图，N是多少个分类，默认是10个，10个人的脸s
        """
        figure = plt.figure(figsize=(5, 5))  # figsize用来设置图形的大小，a为图形的宽， b为图形的高，单位为"英寸"

        logger.debug("Matplot显示数据：%d行", len(features))
        logger.debug("Matplot显示标签：%d个", len(labels))

        # 按照不同的类别，过滤他们，然后画出他们
        for label in range(10):
            # 使用pytorch的tensor来过滤
            # indices = (label == labels).nonzero(as_tuple=True)
            # label_features = torch.index_select(features, 0, indices[0])
            # label_features = label_features.detach().numpy()  # 必须要这么干，按照异常提示里做的

            # 使用numpy的array来过滤
            mask = (label == labels)
            label_features = features[mask]

            plt.scatter(label_features[:, 0], label_features[:, 1])  # 我靠，只显示前2维，高维也只是前2维

        # plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='jpg')
        plt.close(figure)
        buf.seek(0)

        nparray = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(nparray,1)
        if not self.image(image, name, step):
            logger.error("保存Embeding Plot到tensorboad失败：%r", image.shape)

    # https://www.cnblogs.com/cloud-ken/p/9329703.html
    # 生成可视化最终输出层向量所需要的日志文件
    # 暂时不用了，太麻烦，还得搞sprite图啥的
    def plot_tf_embedding(self, features, name, step):
        """
        使用tensorflow的embeding API，直接输出高维向量
        @:param features: List<List<feature[512]>>
            list - 多个人
                list - 一个人的
                    feature 一个人的一张脸的
        """
        for person in features:
            for feature in person:
                # 使用一个新的变量来保存最终输出层向量的结果，
                # 因为embedding是通过Tensorflow中变量完成的，
                # 所以PROJECTOR可视化的都是TensorFlow中的变量，
                # 所以这里需要新定义一个变量来保存输出层向量的取值
                y = tf.Variable(feature, name="face")

                # 通过project.ProjectorConfig类来帮助生成日志文件
                config = projector.ProjectorConfig()
                # 增加一个需要可视化的bedding结果
                embedding = config.embeddings.add()
                # 指定这个embedding结果所对应的Tensorflow变量名称
                embedding.tensor_name = y.name

                # Specify where you find the metadata
                # 指定embedding结果所对应的原始数据信息。
                # 比如这里指定的就是每一张MNIST测试图片对应的真实类别。
                # 在单词向量中可以是单词ID对应的单词。
                # 这个文件是可选的，如果没有指定那么向量就没有标签。
                # embedding.metadata_path = META_FIEL

                # Specify where you find the sprite (we will create this later)
                # 指定sprite 图像。这个也是可选的，如果没有提供sprite 图像，那么可视化的结果
                # 每一个点就是一个小困点，而不是具体的图片。
                # embedding.sprite.image_path = SPRITE_FILE
                # 在提供sprite图像时，通过single_image_dim可以指定单张图片的大小。
                # 这将用于从sprite图像中截取正确的原始图片。
                # embedding.sprite.single_image_dim.extend([28, 28])

                # Say that you want to visualise the embeddings
                # 将PROJECTOR所需要的内容写入日志文件。
                projector.visualize_embeddings(self.summaryWriter, config)

                # 生成会话，初始化新声明的变量并将需要的日志信息写入文件。
                sess = tf.InteractiveSession()
                sess.run(tf.global_variables_initializer())
                saver = tf.train.Saver()
                saver.save(sess, os.path.join("logs", "tboard"), step)
                logger.debug("保存embedding")


class VisdomVisualizer(object):
    """
    如果要使用visdom，需要启动一个服务器，然后连接到这个服务器上，
    vis.check_connection()就是在连接服务器上，
    """

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

    def text(self, iters, x, name='train_loss'):
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
