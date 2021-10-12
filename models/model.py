# -*- coding: utf-8 -*-
"""
Created on 18-5-21 下午5:26

@author: ronghuaiyang
"""
# __all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
#            'resnet152']
import logging

import torch.nn as nn
from torchvision import models

from models.metrics import ArcMarginProduct

logger = logging.getLogger(__name__)


class Net(nn.Module):
    """
           /(2)-10分类 <-- MNist
    512---|
          \10000分类   <-- faces

    参数量（人脸）：resnet50（2345万）+

    """

    def __init__(self, type, device, config):
        super(Net, self).__init__()

        size = config.input_shape[-1]
        kernel_size = size // 32  # resnet最后都是缩小32倍，无论resnet18还是resnet50

        self.is_arcface_metrics = True

        # mnist+啥都不干的交叉熵，把7x7x512，拉平后，降维成2维度（2维度是为了显示embedding到plot用）
        if type == "mnist.ce":
            num_classes = 10
            resnet_model = models.resnet18(pretrained=True)
            self.resnet_layer = nn.Sequential(*list(resnet_model.children())[:-2])
            self.extract_layer = nn.Sequential(
                nn.Linear(kernel_size * kernel_size * 512, 2),  # 5x5x512=25.6万个参数/resnet18是1100万个参数
                nn.BatchNorm1d(2))
            self.metric_fc = nn.Sequential(
                nn.Linear(2, num_classes),
                nn.BatchNorm1d(num_classes))
            logger.info("构建验证MNIST数据集的交叉熵模型")
            self.is_arcface_metrics = False  # 如果不是arcface，不需要处理label
            return

        # mnist+arcface，把7x7x512，拉平后，降维成2维度（2维度是为了显示embedding到plot用）
        if type == "mnist.arcface":
            num_classes = 10
            resnet_model = models.resnet18(pretrained=True)
            self.resnet_layer = nn.Sequential(*list(resnet_model.children())[:-2])
            self.extract_layer = nn.Sequential(
                nn.Linear(kernel_size * kernel_size * 512, 2),  # 5x5x512=25.6万个参数/resnet18是1100万个参数
                nn.BatchNorm1d(2))
            self.metric_fc = ArcMarginProduct(2,
                                              num_classes,
                                              s=30,
                                              m=0.5,
                                              easy_margin=config.easy_margin,
                                              device=device)
            logger.info("构建验证MNIST数据集的Arcface模型")
            return

        if type == "face":
            num_classes = config.num_classes
            resnet_model = models.resnet50(pretrained=True)
            self.resnet_layer = nn.Sequential(*list(resnet_model.children())[:-1])  # 只去掉1层，保持2048的全局池化层avg层

            # 全连接参数：3100万，但是不收敛，保留代码，切换成avg，降低参数量
            # self.resnet_layer = nn.Sequential(*list(resnet_model.children())[:-2])
            # self.extract_layer = nn.Sequential(
            #     nn.Linear(kernel_size * kernel_size * 2048, config.embedding_size),
            #     nn.BatchNorm1d(config.embedding_size))

            self.extract_layer = nn.Sequential(
                nn.Linear(2048, config.embedding_size),
                nn.BatchNorm1d(config.embedding_size))

            self.metric_fc = ArcMarginProduct(config.embedding_size,  # arcface里面包含了weight，类上面的Linear的weight
                                              num_classes,  # 2048是因为resnet50输出是2048通道，resnet18是512
                                              s=64,  # 参考旷视insightface的代码设置
                                              m=1.2,  # 论文建议是m=1.2
                                              easy_margin=True,
                                              device=device)
            logger.info("构建人脸的Arcface模型")
            return

        raise ValueError("无法识别的模型类型：" + type)

    def __num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x, label):
        x = self.resnet_layer(x)
        x = x.view(-1, self.__num_flat_features(x))  # flat it

        # 如果self.extract_2dim_layer is not none， call it
        features = self.extract_layer(x)

        if self.is_arcface_metrics:
            x = self.metric_fc(features, label)
        else:
            x = self.metric_fc(features)
        return x, features

    def extract_feature(self, x):
        x = self.resnet_layer(x)
        x = x.view(-1, self.__num_flat_features(x))  # flat it
        # logger.debug("resnet抽取完的特征：%r",x.shape)
        features = self.extract_layer(x)
        return features

    def predict(self, x):
        features = self.extract_feature(x)

        if self.is_arcface_metrics:
            x = self.metric_fc.cosθ(features)
        else:
            x = self.metric_fc(features)
        return x
