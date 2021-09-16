import argparse
import datetime
import logging
import time

import numpy as np
import torch
from torch.nn import DataParallel
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchsummary import summary

from config.config import Config
from models.focal_loss import FocalLoss
from models.metrics import ArcMarginProduct
from models.resnet import resnet50
from test import test
from utils import init_log
from utils.dataset import Dataset
from utils.early_stop import EarlyStop
from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)


def save_model(opt, epoch, model, train_size, loss, acc):
    # 'checkpoints/arface_{}_{}_{}.model' # epoch,datetime,loss,acc
    step = epoch * train_size
    now = datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path = opt.checkpoints_path.format(epoch, step, now, loss, acc)
    torch.save(model.state_dict(), model_path)
    logger.info("模型保存到：%s", model_path)


def main(args):
    opt = Config()

    # 准备数据
    train_dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
    trainloader = DataLoader(train_dataset,
                             batch_size=opt.train_batch_size,
                             shuffle=True,
                             num_workers=opt.num_workers)
    logger.info('每个Epoch有%d个批次，每个批次%d张', len(trainloader), opt.train_batch_size)
    train_size = None

    # 准备调试参数
    if args.mode == "debug":
        logger.info("启动调试模式 >>>>> ")
        opt.train_batch_size = 1
        opt.max_epoch = 2
        opt.test_batch_size = 1
        opt.test_size = 3
        opt.print_freq = 1
        train_size = 5

    # 准备神经网络
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device代表将torch.Tensor分配到的设备的对象
    logger.info("训练使用:%r", device)
    criterion = FocalLoss(gamma=2)
    model = resnet50(opt, pretrained=True)
    # 你注意这个细节，这个是一个网络中的"层";需要传入num_classes，也就是说，多少个人的人脸就是多少类，这里大概是1万左右（不同数据集不同）
    # 另外第一个入参是输入维度，是512，why？是因为resnet50网络的最后的输出就是512：self.fc5 = nn.Linear(512 * 8 * 8, 512)
    metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin, device=device)
    model.to(device)
    model = DataParallel(model)
    # 为何loss，也需要用这么操作一下？
    metric_fc.to(device)  # 用xxx设备
    metric_fc = DataParallel(metric_fc)  # 走并行模式
    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': metric_fc.parameters()}],
                                 lr=opt.lr, weight_decay=opt.weight_decay)
    scheduler = StepLR(optimizer, step_size=opt.lr_step, gamma=0.1)  # StepLR是调整学习率的
    early_stopper = EarlyStop(opt.early_stop)
    summary(model, opt.input_shape)

    # 其他准备
    visualizer = Visualizer(opt)
    min_loss = 999999
    latest_loss = 99999
    total_steps = 0
    start = time.time()

    for epoch in range(opt.max_epoch):

        scheduler.step()
        model.train()

        epoch_start = time.time()
        for step_of_epoch, data in enumerate(trainloader):

            # 这个是为了测试方便，只截取很短的数据训练
            if train_size and step_of_epoch > train_size:
                logger.info("当前epoch内step[%d] > 训练最大数量[%d]，此epoch提前结束", step_of_epoch, train_size)
                break

            try:
                images, label = data
                images = images.to(device)
                label = label.to(device).long()
                # logger.debug("【训练】训练数据：%r", images.shape)
                # logger.debug("【训练】模型要求输入：%r", list(model.parameters())[0].shape)
                feature = model(images)
                output = metric_fc(feature, label)
                loss = criterion(output, label)
                optimizer.zero_grad()  # 先把所有的梯度都清零？为何？
                loss.backward()
                optimizer.step()
                latest_loss = loss.item()
                # 每隔N个batch，就算一下这个批次的正确率
                if total_steps % opt.print_freq == 0:
                    output = output.data.cpu().numpy()
                    output = np.argmax(output, axis=1)
                    label = label.data.cpu().numpy()
                    train_batch_acc = np.mean((output == label).astype(int))
                    speed = total_steps / (time.time() - start)
                    logger.info("Epoch[%s]耗时%.2f秒,迭代[%d],速度[%.0f步/秒],loss[%.4f],batch_acc[%.4f]",
                                epoch,
                                epoch_start - time.time(),
                                total_steps,
                                speed,
                                loss.item(),
                                train_batch_acc)

                    if visualizer:
                        visualizer.write(total_steps, loss.item(), name='train_loss')
                        visualizer.write(total_steps, train_batch_acc, name='train_acc')
            except:
                logger.exception("训练出现异常，继续...")

            total_steps = epoch * len(trainloader) + step_of_epoch

        # 尝试预测
        acc = -1
        try:
            model.eval()
            acc = test(model, opt)  # <---- 预测
        except:
            logger.exception("验证出现异常，继续...")

        if latest_loss < min_loss:
            logger.info("Epoch[%d] loss[%.4f] 比之前 loss[%.4f] 更低，保存模型",
                        epoch,
                        latest_loss,
                        min_loss)
            min_loss = latest_loss
            save_model(opt, epoch, model, train_size, latest_loss, acc)

        # early_stopper可以帮助存基于acc的best模型
        early_stopper.decide(acc, save_model, opt, epoch + 1, model, train_size, latest_loss, acc)

        if visualizer:
            total_steps = (epoch + 1) * train_size
            visualizer.write(total_steps, acc, name='test_acc')

    logger.info("训练结束，耗时%.2f小时，共%d个epochs，%d步",
                (time.time() - start) / 3600,
                epoch + 1,
                total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--mode", default="normal", type=str)
    args = parser.parse_args()

    logger.info("参数配置：%r", args)

    init_log()

    main(args)
