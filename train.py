from __future__ import print_function

import argparse
import datetime
import logging
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
from test import test
from utils import init_log
from utils.dataset import Dataset
from utils.early_stop import EarlyStop
from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


def save_model(epoch, model, train_size, loss, acc):
    # 'checkpoints/arface_{}_{}_{}.model' # epoch,datetime,loss,acc
    step = epoch * train_size
    now = datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = opt.checkpoints_path.format(epoch, step, now, loss, acc)
    torch.save(model.state_dict(), model_path)
    logger.info("模型保存到：%s", model_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--mode", default="normal", type=str)
    args = parser.parse_args()

    min_loss = 999999

    logger.info("参数配置：%r", args)

    init_log()

    opt = Config()
    if args.mode == "debug":
        logger.info("启动调试模式 >>>>> ")
        opt.max_epoch = 2
        opt.test_batch_size = 1
        opt.test_batch_size = 1
        opt.test_size = 3

    if opt.display:
        visualizer = Visualizer()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device代表将torch.Tensor分配到的设备的对象
    logger.info("训练使用:%r", device)

    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = data.DataLoader(train_dataset,
                                  batch_size=opt.train_batch_size,
                                  shuffle=True,
                                  num_workers=opt.num_workers)

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
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin,device=device)
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
    early_stopper = EarlyStop(opt.early_stop)

    start = time.time()
    train_size = len(trainloader)
    latest_loss = 99999
    for epoch in range(opt.max_epoch):
        scheduler.step()
        model.train()

        for step_of_epoch, data in enumerate(trainloader):
            data_input, label = data
            data_input = data_input.to(device)
            label = label.to(device).long()
            feature = model(data_input)
            output = metric_fc(feature, label)
            loss = criterion(output, label)
            optimizer.zero_grad()  # 先把所有的梯度都清零？为何？
            loss.backward()
            optimizer.step()
            latest_loss = loss.item()

            total_steps = epoch * len(trainloader) + step_of_epoch

            # 每隔N个batch，就算一下这个批次的正确率
            if total_steps % opt.print_freq == 0:
                output = output.data.cpu().numpy()
                output = np.argmax(output, axis=1)
                label = label.data.cpu().numpy()
                train_batch_acc = np.mean((output == label).astype(int))
                speed = opt.print_freq / (time.time() - start)
                time_str = time.asctime(time.localtime(time.time()))
                logger.info("Epoch[%s],迭代[%d],速度[%.0f],loss[%.4f],batch_acc[%.4f]",
                            epoch,
                            total_steps,
                            speed,
                            loss.item(),
                            train_batch_acc)

                if opt.display:
                    visualizer.display_current_results(total_steps, loss.item(), name='train_loss')
                    visualizer.display_current_results(total_steps, acc, name='train_acc')

                start = time.time()

        model.eval()
        acc = test(model, opt)

        if latest_loss < min_loss:
            logger.info("Epoch[%d] loss[%.4f] 比之前 loss[%.4f] 更低，保存模型",
                        epoch,
                        latest_loss,
                        min_loss)
            save_model(epoch, model, train_size, latest_loss, acc)

        # early_stopper可以帮助存基于acc的best模型
        # save_model(epoch, model,train_size,loss,acc):
        early_stopper.decide(acc, save_model, epoch + 1, model, train_size, latest_loss, acc)

        if opt.display:
            total_steps = (epoch + 1) * train_size
            visualizer.display_current_results(total_steps, acc, name='test_acc')
