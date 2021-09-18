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

import utils

logger = logging.getLogger(__name__)


def calculate_features(model, image_paths):
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
        logger.debug("推断实际输出：%r", feature.shape)
        logger.debug("推断实际输出：%r", feature)

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
    face1_face2_label_list = load_test_pairs(opt.lfw_test_pair_path, opt.test_pair_size)
    face_image_names = extract_face_images(face1_face2_label_list)
    face_image_paths = [os.path.join(opt.lfw_root, each) for each in face_image_names]

    s = time.time()
    image_feature_dicts = calculate_features(model, face_image_paths)
    # logger.debug("人脸的特征维度：%r", len(image_feature_dicts))
    t = time.time() - s
    logger.info('[验证]耗时: %.2f秒, 每张耗时：%.4f秒', t, t / len(image_feature_dicts))

    acc, th = test_performance(image_feature_dicts, face1_face2_label_list)
    logger.info("[验证]测试%d对人脸，（最好）正确率%.2f，(适配出来的最好的阈值%.2f)", len(face1_face2_label_list), acc, th)
    return acc


def caculate_samples(model, opt):
    """
    重构后的测试入口，它去加载 形如 "xxx.jpg xxx.jpg 1"的lfw的测试文件，0/1表示是不是同一个人的脸，
    """

    face_dirs = load_samples(opt.lfw_root, opt.test_classes)
    s = time.time()
    different_faces = []
    count = 0

    for face_dir,file_num in face_dirs:
        file_names = os.listdir(face_dir)
        count += len(file_names)
        full_paths = [os.path.join(face_dir, file_name) for file_name in file_names]
        image_feature_dicts = calculate_features(model, full_paths)
        different_faces.append(image_feature_dicts.values())
    t = time.time() - s

    logger.info('[计算%d个人的%d张人脸] 耗时: %.2f秒, 每张耗时：%.4f秒', opt.test_size, count, t, t / len(face_dirs))

    return different_faces


def load_test_pairs(test_file_path, pair_size):
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


def load_samples(dir, size):
    """
    加载测试集中，人脸最多的前N个人，这个用于embedding显示
    """

    dirs = os.listdir(dir)
    dirs = [os.path.join(dir, sub_dir) for sub_dir in dirs]
    dirs = [dir for dir in dirs if os.path.isdir(dir)]
    logger.debug("从目录[%s],加载测试文件夹：%d 个",dir,len(dirs))
    dir_files = {}
    for dir in dirs:
        dir_files[dir] = len(os.listdir(dir))

    sored_dir_files = [[k, v] for k, v in sorted(dir_files.items(), key=lambda item: item[1])]
    #sored_dir_files = sored_dir_files[-size:]
    sored_dir_files = sored_dir_files[:3]
    logger.debug("过滤后，剩余%d个文件夹",len(sored_dir_files))
    return sored_dir_files
