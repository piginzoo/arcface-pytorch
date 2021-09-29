# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function

import logging
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

import utils
from utils.dataset import get_dataset

logger = logging.getLogger(__name__)


def get_tester(type, opt, device):
    # 准备数据，如果mode是"mnist"，使用MNIST数据集
    # 可视化，其实就是使用MNIST数据集，训练一个2维向量
    # mnist数据，用于可视化的测试
    if type == "mnist":
        tester = MnistTester(opt, device)
    else:
        tester = FaceTester()
    return tester


class Tester():
    """
    为了适配2种测试：一种是人脸的，一种是实验用的MNIST（为了可视化）
    """

    def acc(self, model, metric, opt):
        pass

    def calculate_features(self, model, opt):
        pass


class MnistTester(Tester):
    def __init__(self, opt, device):
        dataset = get_dataset(train=False, type='mnist', opt=opt)
        self.data_loader = DataLoader(dataset,
                                      batch_size=32,  # 测试 = 3 | 32
                                      shuffle=True,
                                      num_workers=0)
        self.device = device

    def acc(self, model, opt):
        correct = 0
        start = time.time()
        for index, data in enumerate(self.data_loader):
            imgs_of_batch, label = data
            # bugfix:...found at least two devices, cuda:0 and cpu!
            imgs_of_batch, label = imgs_of_batch.to(self.device), label.to(self.device)

            if index > self.test_size:
                logger.debug("[验证] 测试数据集长度超限：%d，模型计算验证数据结束", index)
                break

            # 预测
            with torch.no_grad():
                output, features = model(imgs_of_batch)

                # 本来还想要再经过一下arcface的metrics，也就是论文的那个s*cos(θ+m)，
                # 但是，突然反思了一下，觉得不对，因为那个是需要同时传入label，我靠，我用网络就是为了argmax得到label，你让我传给你label，什么鬼？
                # 显然我是理解错了，对比看了真实人脸的acc代码，在下面FaceTest.acc的实现里，test_performance方法里，
                # 那个根本没有用metrics（也就是arcface的loss），而是直接用resnet的输出，算出两个不同的x1、x2的夹角，
                # 且通过一堆人脸(6000个）得到一个阈值，来判断是不是同一人脸，人家是在做这事！
                #
                # 而我们这个acc，就是要简单的判断是哪个数字，不是要判断2张图是不是同一数字啊。

                # 我只要看从resnet出来的向量就可以，argmax的那个就是最像的类别（不用softmax了，softmax只是为了放大而已）
                pred = output.max(1, keepdim=True)[1]

            correct += pred.eq(label.view_as(pred)).sum().item()

        acc = correct / (index * self.data_loader.batch_size)
        logger.info("测试了%d条，正确%d条，正确率：%.4f，耗时：%.2f",
                    index * self.data_loader.batch_size,
                    correct, acc, time.time() - start)
        return acc

    def calculate_features(self, model, image_paths):
        features = None
        labels = None
        for data, label in self.data_loader:
            data = data.to(self.device)  # 放到显存中，用于加速
            # you don't need to calculate gradients for forward and backward phase.防止OOM
            with torch.no_grad():
                _, __features = model(x=data)
                __features = __features.cpu()  # 用cpu()值替换掉原引用，导致旧引用回收=>GPU内存回收，解决OOM问题

            if features is None:
                features = __features.numpy()
            else:
                features = np.concatenate((features, __features.numpy()))
            if labels is None:
                labels = label.numpy()
            else:
                labels = np.concatenate((labels, label.numpy()))
        return features, labels


class FaceTester(Tester):

    def calculate_features(self, model, image_paths):
        """
        image_paths: 所有的图片的路径（全路径）
        """

        image_feature_dict = {}
        for i, image_path in enumerate(image_paths):
            name = os.path.split(image_path)[1]
            image = utils.load_image(image_path)
            if image is None: continue

            data = np.array([image])
            data = torch.from_numpy(data)
            # logger.debug("推断要求输入：%r", list(model.parameters())[0].shape)
            logger.debug("推断实际输入：%r", data.shape)
            feature = model(data)[0]
            feature = feature.cpu().numpy()

            logger.debug("推断实际输出：%r", feature.shape)
            logger.debug("推断实际输出：%r", feature)

            image_feature_dict[name] = feature
        return image_feature_dict

    def load_model(self, model, model_path):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    def cosin_metric(self, x1, x2):
        """
          x1 。x2
        --------- ==> 这个就是x1，x2的夹角θ的cos值，arcface的θ，是和权重W_i的角度，不是两张脸的features：x1、x2的角度，
        |x1|*|x2|     但因为模型导致某一类他们都压缩成一个聚集的类了，变相地导致了，不同类之间的角度就很大，
        """
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    def cal_accuracy(self, y_score, y_true):
        """
        从一堆的 脸脸对，得到cosθ差异值，
        然后用每一个cosθ差异值当阈值，来算正确率，
        对应最好的正确率的那个阈值，当做最好的阈值。
        这个算法有点意思。

        y_score: 是x1，x2之间的夹角θ的cosθ值，
        一共有多少个呢？有很多个，一对就是一个，我记得测试数据有6000个，3000个同一个人的脸，3000个不同人的脸，
        """

        y_score = np.asarray(y_score)
        y_true = np.asarray(y_true)
        best_acc = 0
        best_th = 0
        for i in range(len(y_score)):
            th = y_score[i]
            # y_score是夹角的余弦值，你可以理解成一个夹角，大于这个值（也就是小于某个夹角，余弦是递减的），就是同一人
            # 然后，我用所有的比对的cos值，都试验一遍，效果最好的那个值（也就是某个夹角角度），就是最好的阈值
            y_test = (y_score >= th)
            acc = np.mean((y_test == y_true).astype(int))
            if acc > best_acc:
                best_acc = acc
                best_th = th

        return (best_acc, best_th)

    def test_performance(self, feature_dict, pairs):
        sims = []
        labels = []
        for face1, face2, label in pairs:

            feature_1 = feature_dict.get(face1, None)
            feature_2 = feature_dict.get(face2, None)
            if feature_1 is None or feature_2 is None:
                continue

            sim = self.cosin_metric(feature_1, feature_2)  # 计算cosθ
            sims.append(sim)
            labels.append(label)

        acc, th = self.cal_accuracy(sims, labels)
        return acc, th

    def extract_face_images(self, face1_face2_label_list):
        face_image_paths = []
        for face1, face2, _ in face1_face2_label_list:
            if face1 not in face_image_paths:
                face_image_paths.append(face1)
            if face1 not in face_image_paths:
                face_image_paths.append(face2)
        return face_image_paths

    def acc(self, model, opt):
        """
        重构后的测试入口，它去加载 形如 "xxx.jpg xxx.jpg 1"的lfw的测试文件，0/1表示是不是同一个人的脸，
        """
        face1_face2_label_list = self.load_test_pairs(opt.lfw_test_pair_path, opt.test_pair_size)
        face_image_names = self.extract_face_images(face1_face2_label_list)
        face_image_paths = [os.path.join(opt.lfw_root, each) for each in face_image_names]

        s = time.time()
        image_feature_dicts = self.calculate_features(model, face_image_paths)
        # logger.debug("人脸的特征维度：%r", len(image_feature_dicts))
        t = time.time() - s
        logger.info('[验证]耗时: %.2f秒, 每张耗时：%.4f秒', t, t / len(image_feature_dicts))

        acc, th = self.test_performance(image_feature_dicts, face1_face2_label_list)
        logger.info("[验证]测试%d对人脸，（最好）正确率%.2f，(适配出来的最好的阈值%.2f)", len(face1_face2_label_list), acc, th)
        return acc

    def caculate_samples(self, model, opt):
        """
        重构后的测试入口，它去加载 形如 "xxx.jpg xxx.jpg 1"的lfw的测试文件，0/1表示是不是同一个人的脸，
        """

        # 仅装载前10个人的脸
        face_dirs = self.load_samples(opt.lfw_root, opt.test_classes)
        start = time.time()
        different_faces = []
        count = 0

        for face_dir, file_num in face_dirs:
            # 一个人脸文件夹，包含多个人脸
            file_names = os.listdir(face_dir)
            count += len(file_names)
            full_paths = [os.path.join(face_dir, file_name) for file_name in file_names]
            image_feature_dicts = self.calculate_features(model, full_paths)
            different_faces.append(list(image_feature_dicts.values()))  # 只保留value数据，多个人脸
        t = time.time() - start

        logger.info('[计算%d个人的%d张人脸] 耗时: %.2f秒, 每张耗时：%.4f秒', opt.test_size, count, t, t / len(face_dirs))

        return different_faces

    def load_test_pairs(self, test_file_path, pair_size):
        """
        各加载pair_size的一半的比较对
        """

        fd = open(test_file_path, 'r')
        lines = fd.readlines()
        fd.close()

        random.shuffle(lines)  # shuffle一下

        positive_list = []
        negtive_list = []
        face1_face2_label_list = []

        half_size = pair_size // 2

        for line in lines:
            line = line.strip()
            pairs = line.split()
            face1 = pairs[0]
            face2 = pairs[1]
            label = int(pairs[2])
            if label == 1:
                positive_list.append([face1, face2, label])
            if label == 0:
                negtive_list.append([face1, face2, label])

        face1_face2_label_list += positive_list[:half_size]
        face1_face2_label_list += negtive_list[:half_size]

        logger.info("从[%s]加载比较对%d个", test_file_path, len(face1_face2_label_list))
        return face1_face2_label_list

    def load_samples(self, dir, size):
        """
        加载测试集中，人脸最多的前N个人，这个用于embedding显示
        """

        dirs = os.listdir(dir)
        dirs = [os.path.join(dir, sub_dir) for sub_dir in dirs]
        dirs = [dir for dir in dirs if os.path.isdir(dir)]
        logger.debug("从目录[%s],加载测试文件夹：%d 个", dir, len(dirs))
        dir_files = {}
        for dir in dirs:
            dir_files[dir] = len(os.listdir(dir))

        sored_dir_files = [[k, v] for k, v in sorted(dir_files.items(), key=lambda item: item[1])]
        # sored_dir_files = sored_dir_files[-size:]
        sored_dir_files = sored_dir_files[:3]
        logger.debug("过滤后，剩余%d个文件夹", len(sored_dir_files))
        return sored_dir_files
