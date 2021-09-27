import argparse
import datetime
import logging
import math
import time

import numpy as np
import torch
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchsummary import summary

from config.config import Config
from models import get_resnet
from models.metrics import ArcMarginProduct
from test import MnistTester, FaceTester
from utils import init_log
from utils.dataset import Dataset
from utils.dataset import get_mnist_dataset
from utils.early_stop import EarlyStop
from utils.visualizer import TensorboardVisualizer

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
    train_size = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device代表将torch.Tensor分配到的设备的对象

    # 准备数据，如果mode是"mnist"，使用MNIST数据集
    # 可视化，其实就是使用MNIST数据集，训练一个2维向量
    # mnist数据，用于可视化的测试
    if args.mode == "mnist":
        logger.info("训练MNIST数据 >>>>> ")
        dataset = get_mnist_dataset(True, opt)
        tester = MnistTester(opt, device)

        # 测试
        # opt.max_epoch = 3
        # train_size = 3
    else:
        # 正常的人脸数据
        dataset = Dataset(opt.train_root, opt.train_list, phase='train', input_shape=opt.input_shape)
        tester = FaceTester()
    trainloader = DataLoader(dataset,
                             batch_size=opt.train_batch_size,
                             shuffle=True,
                             num_workers=opt.num_workers)
    logger.info('每个Epoch有%d个批次，每个批次%d张', len(trainloader), opt.train_batch_size)

    # 准备调试参数
    if args.mode == "debug":
        logger.info("启动调试模式 >>>>> ")
        opt.train_batch_size = 3
        opt.max_epoch = 3
        opt.test_batch_size = 1
        opt.test_size = 3
        opt.print_batch = 1
        opt.test_pair_size = 6
        train_size = 5

    # 准备神经网络
    logger.info("训练使用:%r", device)
    criterion = torch.nn.CrossEntropyLoss()  # FocalLoss(gamma=2)
    model = get_resnet(opt, args.mode)

    # 你注意这个细节，这个是一个网络中的"层";需要传入num_classes，也就是说，多少个人的人脸就是多少类，这里大概是1万左右（不同数据集不同）
    # 另外第一个入参是输入维度，是512，why？是因为resnet50网络的最后的输出就是512：self.fc5 = nn.Linear(512 * 8 * 8, 512)
    if args.mode == "mnist":
        # 可视化要求最后输出的维度不是512，而是2，是512之后再接个2
        # metric_fc = ArcMarginProduct(in_features=2, out_features=10, s=30, m=0.5, easy_margin=opt.easy_margin, device=device)
        metric_fc = torch.nn.Linear(2,10)
    else:
        metric_fc = ArcMarginProduct(512, opt.num_classes, s=30, m=0.5, easy_margin=opt.easy_margin, device=device)

    model.to(device)
    model = DataParallel(model)
    # 为何loss，也需要用这么操作一下？
    metric_fc.to(device)  # 用xxx设备
    metric_fc = DataParallel(metric_fc)  # 走并行模式
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': metric_fc.parameters()}],
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)
    early_stopper = EarlyStop(opt.early_stop)

    # 为了打印网络结构，需要传入一个input的shape
    summary(model, opt.input_shape)

    # 其他准备
    visualizer = TensorboardVisualizer(opt.tensorboard_dir)
    min_loss = 999999
    latest_loss = 99999
    total_steps = 0
    start = time.time()

    for epoch in range(opt.max_epoch):

        model.train()

        epoch_start = time.time()
        for step_of_epoch, data in enumerate(trainloader):

            # 这个是为了测试方便，只截取很短的数据训练
            if train_size and step_of_epoch > train_size:
                logger.info("当前epoch内step[%d] > 训练最大数量[%d]，此epoch提前结束", step_of_epoch, train_size)
                break
            total_steps = total_steps + 1

            try:
                images, label = data

                images = images.to(device)
                label = label.to(device).long()

                if np.isnan(images).any():
                    logger.error("图片数据出现异常：epoch:%d/step:%d", epoch, step_of_epoch)
                    continue
                logger.debug("图像数据：%r",images)

                logger.debug("【训练】训练数据：%r", images.shape)
                logger.debug("【训练】模型要求输入：%r", list(model.parameters())[0].shape)
                feature = model(images)
                logger.debug("【训练】训练输出features：%r", feature)
                output = metric_fc(feature, label)  #
                logger.debug("【训练】训练输出output：%r", feature)
                loss = criterion(output, label)

                # 以SGD为例，是算一个batch计算一次梯度，然后进行一次梯度更新。这里梯度值就是对应偏导数的计算结果。
                # 我们进行下一次batch梯度计算的时候，前一个batch的梯度计算结果，没有保留的必要了。
                # 所以在下一次梯度更新的时候，先使用optimizer.zero_grad把梯度信息设置为0。
                optimizer.zero_grad()

                loss.backward()

                # 做梯度裁剪
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

                optimizer.step()

                latest_loss = loss.item()
                if math.isnan(latest_loss):
                    logger.error("训练过程中出现loss为nan, epoch[%d], step[%d]", epoch, step_of_epoch)

                # 每隔N个batch，就算一下这个批次的正确率
                if total_steps % opt.print_batch == 0:
                    logger.debug("[可视化] 第%d批", total_steps)

                    # 从tensor=>numpy(device从cuda=>cpu)
                    output = output.cpu().detach().numpy()  # 1.cpu:复制一份到GPU=>内存 2.detach,去掉梯度，12后才能numpy
                    label = label.cpu().detach().numpy()
                    images = images.cpu().detach().numpy()

                    output = np.argmax(output, axis=1)
                    train_batch_acc = np.mean((output == label).astype(int))
                    speed = total_steps / (time.time() - start)
                    logger.info("[可视化] Epoch[%s]/迭代[%d],耗时%.2f秒,速度[%.0f步/秒],loss[%.4f],batch_acc[%.4f]",
                                epoch,
                                total_steps,
                                time.time() - epoch_start,
                                speed,
                                loss.item(),
                                train_batch_acc)
                    visualizer.text(total_steps, loss.item(), name='train_loss')
                    visualizer.text(total_steps, train_batch_acc, name='train_acc')
                    visualizer.image(images, name="train_images")
            except:
                logger.exception("训练出现异常，继续...")

        # 尝试预测
        acc = -1
        try:
            model.eval()
            acc = tester.acc(model, metric_fc, opt)  # <---- 预测
        except:
            logger.exception("验证出现异常，继续...")

        if latest_loss < min_loss:
            logger.info("Epoch[%d] loss[%.4f] 比之前 loss[%.4f] 更低，保存模型",
                        epoch,
                        latest_loss,
                        min_loss)
            min_loss = latest_loss
            save_model(opt, epoch, model, len(trainloader), latest_loss, acc)

        # early_stopper可以帮助存基于acc的best模型
        if early_stopper.decide(acc, save_model, opt, epoch + 1, model, len(trainloader), latest_loss, acc):
            logger.info("早停导致退出：epoch[%d] acc[%.4f]", epoch + 1, acc)
            break

        logger.info("Epoch [%d] 结束可视化(保存softmax可视化)", epoch)
        total_steps = (epoch + 1) * len(trainloader)
        visualizer.text(total_steps, acc, name='test_acc')

        if args.mode == "mnist":
            features, labels = tester.calculate_features(model, opt)
            logger.debug("计算完的 [%d] 个人脸features", len(features))
            visualizer.plot_2d_embedding(name='classes', features=features, labels=labels, step=total_steps)

    logger.info("训练结束，耗时%.2f小时，共%d个epochs，%d步",
                (time.time() - start) / 3600,
                epoch + 1,
                total_steps)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--mode", default="normal", type=str)  # normal|
    args = parser.parse_args()

    logger.info("参数配置：%r", args)

    init_log()

    main(args)
