# -*- coding: utf-8 -*-
"""
Created on 18-5-30 下午4:55

@author: ronghuaiyang
"""
from __future__ import print_function

import logging
import os
import time

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


def load_image(image_path):

    if not os.path.exists(image_path):
        logger.warning("图片路径不存在：%s", image_path)
        return None

    # 加载成黑白照片
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        logger.warning("图片加载失败：%s", image_path)
        return None

    # 做resize、变黑白（前面做了）、归一化3件事
    # resize
    image = cv2.resize(image, (128, 128))
    image = np.dstack((image, np.fliplr(image))) # fliplr函数将矩阵进行左右翻转
    image = image.transpose((2, 0, 1))
    image = image[:, np.newaxis, :, :]
    image = image.astype(np.float32, copy=False)
    # 归一化
    image -= 127.5
    image /= 127.5
    return image


def calculate_features(model, image_paths):
    image_feature_dict = {}
    for i, image_path in enumerate(image_paths):
        name = os.path.split(image_path)[1]
        image = load_image(image_path)
        if image is None: continue

        # data = torch.from_numpy(np.array([image]))
        data = torch.from_numpy(image)
        # data = data.to(torch.device("cuda"))
        feature = model(data)[0]
        feature = feature.data.cpu().numpy()
        image_feature_dict[name] = feature
    return image_feature_dict


def load_model(model, model_path):
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)


def cosin_metric(x1, x2):
    return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))


def cal_accuracy(y_score, y_true):
    """
    从一堆的 脸脸对，得到cos差异值，
    然后用每一个cos差异值当阈值，来算正确率，
    对应最好的正确率的那个阈值，当做最好的阈值。
    这个算法有点意思。
    """

    y_score = np.asarray(y_score)
    y_true = np.asarray(y_true)
    best_acc = 0
    best_th = 0
    for i in range(len(y_score)):
        th = y_score[i]
        y_test = (y_score >= th)
        acc = np.mean((y_test == y_true).astype(int))
        if acc > best_acc:
            best_acc = acc
            best_th = th

    return (best_acc, best_th)


def test_performance(feature_dict, pairs):
    sims = []
    labels = []
    for face1, face2, label in pairs:

        feature_1 = feature_dict.get(face1, None)
        feature_2 = feature_dict.get(face2, None)
        if feature_1 is None or feature_2 is None:
            continue

        sim = cosin_metric(feature_1, feature_2)  # 计算cosθ
        sims.append(sim)
        labels.append(label)

    acc, th = cal_accuracy(sims, labels)
    return acc, th


def extract_face_images(face1_face2_label_list):
    face_image_paths = []
    for face1, face2, _ in face1_face2_label_list:
        if face1 not in face_image_paths:
            face_image_paths.append(face1)
        if face1 not in face_image_paths:
            face_image_paths.append(face2)
    return face_image_paths


def test(model, opt):
    """
    重构后的测试入口，它去加载 形如 "xxx.jpg xxx.jpg 1"的lfw的测试文件，0/1表示是不是同一个人的脸，
    """
    face1_face2_label_list = load_test_pairs(opt.lfw_test_pair_path)
    face1_face2_label_list = face1_face2_label_list[:opt.test_size]

    face_image_names = extract_face_images(face1_face2_label_list)
    face_image_paths = [os.path.join(opt.lfw_root, each) for each in face_image_names]

    s = time.time()
    image_feature_dicts = calculate_features(model, face_image_paths)
    logger.debug("人脸的特征shape：%r", len(image_feature_dicts))
    t = time.time() - s
    logger.info('耗时: {}, 每张耗时：{}'.format(t, t / len(image_feature_dicts)))

    acc, th = test_performance(image_feature_dicts, face1_face2_label_list)
    logger.info("测试%d对人脸，（最好）正确率%.2f，(适配出来的最好的阈值%.2f)", len(face1_face2_label_list), acc, th)
    return acc


def load_test_pairs(test_file_path):
    face1_face2_label_list = []
    with open(test_file_path, 'r') as fd:
        lines = fd.readlines()
        for line in lines:
            line = line.strip()
            pairs = line.split()
            face1 = pairs[0]
            face2 = pairs[1]
            label = int(pairs[2])
            face1_face2_label_list.append([face1, face2, label])
    return face1_face2_label_list
