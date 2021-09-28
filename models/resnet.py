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


def get_resnet(config, mode):
    if config.backbone == "resnet18":
        model = models.resnet18(pretrained=True)
    elif config.backbone == "resnet50":
        model = models.resnet50(pretrained=True)
        model.layer4
    else:
        logger.error("无法识别的Resnet类型：%s", config.backbone)
        return None

    logger.info("使用预训练的模型Resnet: %s", config.backbone)
    return Net(model, mode)


class Net(nn.Module):
    def __init__(self, model, mode):
        super(Net, self).__init__()

        # 取掉model的后两层：全局平均池化 和 FC
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])

        if mode == "mnist":
            self.fc = nn.Sequential(
                nn.Linear(5 * 5 * 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU())
            logger.info("构建可视化用模型（输出2维）")
        else:
            self.fc = nn.Sequential(
                nn.Linear(5 * 5 * 512, 512),
                nn.BatchNorm1d(512),
                nn.ReLU())
            logger.info("构建正常模型（输出512维）")

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        x = self.resnet_layer(x)
        x = x.view(-1, self.num_flat_features(x))  # flat it
        x = self.fc(x)

        return x
