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
    """

    def __init__(self, type, device, config):
        super(Net, self).__init__()

        if type.startswith("mnist"):
            logger.info("使用预训练的模型Resnet18")
            resnet_model = models.resnet18(pretrained=True)
            num_classes = 10
        else:
            logger.info("使用预训练的模型Resnet50")
            num_classes = config.num_classes
            resnet_model = models.resnet50(pretrained=True)

        # 取掉model的后两层：全局平均池化 和 FC
        self.resnet_layer = nn.Sequential(*list(resnet_model.children())[:-2])

        size = config.input_shape[-1]
        kernel_size = size // 32  # resnet最后都是缩小32倍，无论resnet18还是resnet50

        self.extract_2dim_layer = None
        if type == "mnist.ce":
            self.extract_layer = nn.Sequential(
                nn.Linear(kernel_size * kernel_size * 512, 2),  # 163万个参数/resnet18是1100万个参数
                nn.BatchNorm1d(2))
            self.metric_fc = nn.Sequential(
                nn.Linear(2, num_classes),
                nn.BatchNorm1d(num_classes))
            logger.info("构建验证MNIST数据集的交叉熵模型")
        if type == "mnist.arcface":
            self.extract_layer = nn.Sequential(
                nn.Linear(kernel_size * kernel_size * 512, 2),  # 163万个参数/resnet18是1100万个参数
                nn.BatchNorm1d(2))
            self.metric_fc = ArcMarginProduct(64,
                                              num_classes,
                                              s=30,
                                              m=0.5,
                                              easy_margin=config.easy_margin,
                                              device=device)
            logger.info("构建验证MNIST数据集的Arcface模型")
        else:
            self.extract_layer = nn.Identity()
            # arcface里面包含了weight，类上面的Linear的weight
            self.metric_fc = ArcMarginProduct(512,
                                              num_classes,
                                              s=30,
                                              m=0.5,
                                              easy_margin=config.easy_margin,
                                              device=device)
            logger.info("构建人脸的Arcface模型")

    def __num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(-1, self.__num_flat_features(x))  # flat it

        # 如果self.extract_2dim_layer is not none， call it
        features = self.extract_layer(x)

        x = self.metric_fc(features)
        return x, features
