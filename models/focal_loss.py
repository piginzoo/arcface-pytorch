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
        # ce = -log(p) 参：https://zhuanlan.zhihu.com/p/27223959
        ce = self.ce(input, target)

        p = torch.exp(-ce) # p = e ^ (-ce)

        # 参：https://blog.csdn.net/u014311125/article/details/109470137
        # FL(p)=-alpha (1-p)^gamma * log(p)：
        #      = alpha (1-p)^gamma * (-log(p))
        #      = alpha (1-p)^gamma * ce
        # alpha=1
        #      = (1-p)^gamma * ce
        # 注意最前面的负号没有了，隐含到ce中了，因为ce=-log(p)
        loss = (1 - p) ** self.gamma * ce

        return loss.mean()