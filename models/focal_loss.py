# -*- coding: utf-8 -*-
"""
Created on 18-6-7 上午10:11

@author: ronghuaiyang
"""

import torch
import torch.nn as nn


class FocalLoss(nn.Module):
    """
    https://zhuanlan.zhihu.com/p/49981234
    Focal Loss解决2个问题：
    1、解决样本不均衡问题，用alpha来平衡
    2、解决样本难分/易分样本的调节，用gamma来平衡
    FL(p)=-alpha (1-p)^gamma * log(p)
    """

    def __init__(self, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss()

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()