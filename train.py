from __future__ import print_function

import argparse
import logging
import os
import time

import numpy as np
import torch
from torch import nn  # StepLR是调整学习率的
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils import data

from config.config import Config
from models.focal_loss import FocalLoss
from models.metrics import AddMarginProduct, ArcMarginProduct, SphereProduct
from models.resnet import resnet_face18, resnet34, resnet50
from test import lfw_test, get_lfw_list
from utils import init_log
from utils.dataset import Dataset
from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


def save_model(model, save_path, name, iter_cnt):
    save_name = os.path.join(save_path, name + '_' + str(iter_cnt) + '.pth')
    torch.save(model.state_dict(), save_name)
    return save_name


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch", default=1, type=int)
    args = parser.parse_args()

    logger.info("参数配置：%r",args)

    init_log()

    opt = Config()
    if opt.display:
        visualizer = Visualizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device代表将torch.Tensor分配到的设备的对象
    logger.info("训练使用:%r", device)

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  # batch_size=opt.train_batch_size,
                                  batch_size=args.batch,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

    identity_list = get_lfw_list(opt.lfw_test_list)  # 我理解是加载测试集，为何不用dataset+dataloader?
    img_paths = [os.path.join(opt.lfw_root, each) for each in identity_list]

    print('{} train iters per epoch:'.format(len(trainloader)))

    if opt.loss == 'focal_loss':
        criterion = FocalLoss(gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if opt.backbone == 'resnet18':
        model = resnet_face18(use_se=opt.use_se)
    elif opt.backbone == 'resnet34':
        model = resnet34()
    elif opt.backbone == 'resnet50':
        model = resnet50()

    if opt.metric == 'add_margin':
        metric_fc = AddMarginProduct(512, opt.num_classes, s=30, m=0.35)
    elif opt.metric == 'arc_margin':
        # easy_margin = False
        # 你注意这个细节，这个是一个网络中的"层";需要传入num_classes，也就是说，多少个人的人脸就是多少类
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin)
    elif opt.metric == 'sphere':
        metric_fc = SphereProduct(512, opt.num_classes, m=4)
    else:
        metric_fc = nn.Linear(512, opt.num_classes)

    # view_model(model, opt.input_shape)
    model.to(device)
    model = DataParallel(model)
    # 为何loss，也需要用这么操作一下？
    metric_fc.to(device)  # 用xxx设备
    metric_fc = DataParallel(metric_fc)  # 走并行模式

    if opt.optimizer == 'sgd':
        optimizer = torch.optim.SGD([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                    lr=opt.lr, weight_decay=opt.weight_decay)
    else:
        optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                     lr=opt.lr, weight_decay=opt.weight_decay)
    # StepLR是调整学习率的
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)

    start = time.time()
    #for i in range(opt.max_epoch):
    for i in range(args.epochs):
        scheduler.step()  # ？？？

        model.train()
        for ii, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()  # 先把所有的梯度都清零？为何？
            loss.backward()
            optimizer.step()

            iters = i * len(trainloader) + ii

            if iters % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                # print(output)
                # print(label)
                acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                print('{} train epoch {} iter {} {} iters/s loss {} acc {}'.format(time_str, i, ii, speed, loss.item(),
                                                                                   acc))
                if opt.display:
                    visualizer.display_current_results(iters, loss.item(), name='train_loss')
                    visualizer.display_current_results(iters, acc, name='train_acc')

                start = time.time()

        if i % opt.save_interval == 0 or i == opt.max_epoch:
            save_model(model, opt.checkpoints_path, opt.backbone, i)

        model.eval()
        acc = lfw_test(model, img_paths, identity_list, opt.lfw_test_list, opt.test_batch_size)
        if opt.display:
            visualizer.display_current_results(iters, acc, name='test_acc')
