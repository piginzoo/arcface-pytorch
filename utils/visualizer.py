import io
import logging
import os
import warnings
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import visdom
from PIL import Image
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

    def text(self, step, value, name):
        self.visualizer.write(step, value, name)


class TensorboardVisualizer(object):
    """
    参考：https://jishuin.proginn.com/p/763bfbd5447c
    """

    def __init__(self, log_dir):
        __log_dir = os.path.join(log_dir, datetime.strftime(datetime.now(), '%Y%m%d-%H%M'))

        if not os.path.exists(__log_dir):
            os.makedirs(__log_dir)
        self.summaryWriter = tf.summary.create_file_writer(logdir=__log_dir)  # tf1.x:SummaryWriter(log_dir=__log_dir)
        self.log_dir = __log_dir

    def text(self, step, value, name):
        # for tensorflow1.x，代码保留
        # self.summaryWriter.add_scalar(tag=name, scalar_value=value, global_step=step)
        with self.summaryWriter.as_default():
            tf.summary.scalar(name, value, step=step)

    def image(self, images, name):
        with self.summaryWriter.as_default():
            images = np.transpose(images, (0, 2, 3, 1))  # [B,C,H,W]=>[B,H,W,C], tf2.x的image通道顺序
            tf.summary.image(name, images, 0)  # step=0, 只保留当前批次就可以
            logger.info("保存图保存到tensorboad: %r", images.shape)

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

            plt.scatter(label_features[:, 0], label_features[:, 1])

        # plt.show()
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        buf.seek(0)
        image = np.array(Image.open(buf))
        image_string = buf.getvalue()
        height, width, channel = image.shape

        # for tensorflow1.x，代码保留
        # image = tf.Summary.Image(height=height, width=width, colorspace=channel, encoded_image_string=image_string)
        # summary = tf.Summary(value=[tf.Summary.Value(tag=name, image=image)])
        # writer = tf.summary.FileWriter(self.log_dir)
        # writer.add_summary(summary, step)
        # writer.close()

        # for tensorflow2.x
        with self.summaryWriter.as_default():
            image = tf.image.decode_png(buf.getvalue(), channels=4)
            image = tf.expand_dims(image, 0)
            tf.summary.image(name, image, step=step)
            logger.info("保存Embeding图保存到tensorboad: %r", image.shape)

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
