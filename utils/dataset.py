import logging
import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as T, datasets  # PyTorch框架中有一个非常重要且好用的包
from torchvision.transforms import transforms

logger = logging.getLogger(__name__)


def get_mnist_dataset(train,opt):
    dataset = datasets.MNIST('./data',
                             train=train,
                             download=True,
                             transform=transforms.Compose([
                                 transforms.Resize(opt.input_shape[1:]),
                                 transforms.ToTensor(),
                                 transforms.Lambda(lambda x: x.repeat(3, 1, 1))]))  # gray->rgb
                                 # transforms.Normalize((0.1307,), (0.3081,))]))
    return dataset

class Dataset(data.Dataset):
    """
    类teras的sequence，好熟悉那：https://pytorch.apachecn.org/docs/1.4/5.html
    但是，__getitem__，不负责batch产生，只产生1个，
    那batch谁控制？
    是另外一个torch的类：DataLoader，他有batch,worker,shuffle参数，一看就明白是干啥的了

    所以，Dataset重要的工作，就是如何加载一条数据
    """

    def __init__(self, root, data_list_file, phase, input_shape):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            img_labels = fd.readlines()

        img_labels = [os.path.join(root, img[:-1]) for img in img_labels]

        filtered_imgs = []
        for __img_path_label in img_labels:
            __img_path, label = __img_path_label.split()
            if not os.path.exists(__img_path):
                # logger.debug("%s 不存在",__img_path)
                continue
            filtered_imgs.append([__img_path, label])
        logger.debug("过滤后，共加载图片%d张", len(filtered_imgs))

        self.imgs = np.random.permutation(filtered_imgs)  # 打乱顺序

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        if self.phase == 'train':
            # torchvision:PyTorch框架中有一个非常重要且好用的包
            self.transforms = T.Compose([  # 做增强
                T.RandomCrop(self.input_shape[1:]),  #
                T.RandomHorizontalFlip(),  # 把图片水平方向翻过来，有点像镜子里看
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path, label = self.imgs[index]  # 这里面是图片路径
        image = Image.open(img_path)
        logger.debug("加载图片：%s", img_path)
        image = self.transforms(image)
        label = np.int32(label)
        logger.debug("训练数据：%r", image.shape)
        return image.float(), label

    def __len__(self):
        return len(self.imgs)
