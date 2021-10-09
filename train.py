import argparse
import datetime
import logging
import math
import time

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader
from torchsummary import summary

import test
from config.config import Config
from models import Net
from utils import dataset as data_loader
from utils import init_log
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # torch.device代表将torch.Tensor分配到的设备的对象

    # 测试类
    tester = test.get_tester(args.mode, opt, device)

    # 训练数据加载器
    dataset = data_loader.get_dataset(train=True, type=args.mode, opt=opt)
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

    # 准备神经网络
    logger.info("训练使用:%r", device)
    criterion = torch.nn.CrossEntropyLoss()  # FocalLoss(gamma=2)

    # 创建模型
    model = Net(args.mode, device, opt)
    model.to(device)

    # 创建优化器
    optimizer = torch.optim.Adam([{
        'params': model.parameters(),
        'lr': 0.001
    }])  # 因为是微调，所以设成小一些的0.001，否则从头开始的话，一般都是设成0.1

    early_stopper = EarlyStop(opt.early_stop)

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

            total_steps = total_steps + 1

            # if total_steps>1: break
            try:
                # 准备batch数据/标签
                images, label = data
                # logger.debug("图像数据：%r",images)
                if np.isnan(images.numpy()).any():
                    logger.error("图片数据出现异常：epoch:%d/step:%d", epoch, step_of_epoch)
                    continue
                images = images.to(device)
                label = label.to(device).long()

                # 使用Resnet抽取特征
                output, _ = model(images,label)

                # 求loss
                loss = criterion(output, label)

                # 以SGD为例，是算一个batch计算一次梯度，然后进行一次梯度更新。这里梯度值就是对应偏导数的计算结果。
                # 我们进行下一次batch梯度计算的时候，前一个batch的梯度计算结果，没有保留的必要了。
                # 所以在下一次梯度更新的时候，先使用optimizer.zero_grad把梯度信息设置为0。
                optimizer.zero_grad()

                # 反向梯度下降
                loss.backward()

                # 做梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)

                # 优化器
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
                    train_batch_acc = np.float(np.mean((output == label).astype(int)))
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
            acc = tester.acc(model, opt)  # <---- 预测
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

        if args.mode.startswith("mnist"):
            features_2d, labels = tester.calculate_features(model, opt.lfw_test_pair_path)
            logger.debug("计算完的 [%d] 个人脸features", len(features_2d))

            # 做一个归一化操作
            features_2d = normalize(features_2d, axis=1)
            visualizer.plot_2d_embedding(name='classes', features=features_2d, labels=labels, step=total_steps)

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
