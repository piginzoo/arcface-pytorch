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

logger = logging.getLogger(__name__)


def get_resnet(config,mode):
    if config.backbone == "resnet18":
        model = models.resnet18(pretrained=True)
    elif config.backbone == "resnet50":
        model = models.resnet50(pretrained=True)
    else:
        logger.error("无法识别的Resnet类型：%s", config.backbone)
        return None

    logger.info("使用预训练的模型Resnet: %s", config.backbone)

    # resnet仅有一个全连接层
    # 得到该全连接层输入神经元数.in_features
    fc_features = model.fc.in_features

    # 默认的输出神经元数为1000，修改为自己想输出的人脸的向量，类别为2,即man和woman
    # fc输入为 2048x1x1=2048,新fc输出为512，参数量为1048576，100万个参数
    # 本来我想直接用1000个分类的概率向量再接一个512的全链接呢，但是觉得2个全连接不好收敛，作罢
    # 还是用经典的方式把，也是网上推荐的替换FC的。
    # 这里有个细节：resnet50是把原图缩小到32倍，但是最后却成了1x1，原因是用了全局平均池化的缘故，这块也不动了，
    # 所以无论你什么尺寸，最终都会被平均池化成1x1
    if mode=="visualize":
        model.fc = nn.Sequential(
            nn.Linear(fc_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.BatchNorm1d(2),
            nn.ReLU())
        logger.info("构建可视化用模型（输出2维）")
    else:
        model.fc = nn.Sequential(
            nn.Linear(fc_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU())
        logger.info("构建正常模型（输出512维）")

    return model
