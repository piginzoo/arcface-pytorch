import os

import cv2
import numpy as np
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms as T # PyTorch框架中有一个非常重要且好用的包


class Dataset(data.Dataset):
    """
    类teras的sequence，好熟悉那：https://pytorch.apachecn.org/docs/1.4/5.html
    但是，__getitem__，不负责batch产生，只产生1个，
    那batch谁控制？
    是另外一个torch的类：DataLoader，他有batch,worker,shuffle参数，一看就明白是干啥的了

    所以，Dataset重要的工作，就是如何加载一条数据
    """

    def __init__(self, root, data_list_file, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape

        with open(os.path.join(data_list_file), 'r') as fd:
            imgs = fd.readlines()

        imgs = [os.path.join(root, img[:-1]) for img in imgs]
        self.imgs = np.random.permutation(imgs) # 打乱顺序

        # normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
        #                         std=[0.5, 0.5, 0.5])

        normalize = T.Normalize(mean=[0.5], std=[0.5]) # ????，不知道normal成啥样了

        if self.phase == 'train':
            # torchvision:PyTorch框架中有一个非常重要且好用的包
            self.transforms = T.Compose([ # 感觉是做了增强，不细看了
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
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
        sample = self.imgs[index] # 这里面是图片路径
        splits = sample.split()
        img_path = splits[0]
        data = Image.open(img_path) # 加载图片
        data = data.convert('L') # a greyscale ("L") ，L是灰度图像
        data = self.transforms(data)
        label = np.int32(splits[1])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(root='/data/Datasets/fv/dataset_v1.1/dataset_mix_aligned_v1.1',
                      data_list_file='/data/Datasets/fv/dataset_v1.1/mix_20w.txt',
                      phase='sandbox',
                      input_shape=(1, 128, 128))

    trainloader = data.DataLoader(dataset, batch_size=10)
    for i, (data, label) in enumerate(trainloader):
        # imgs, labels = data
        # print imgs.numpy().shape
        # print data.cpu().numpy()
        # if i == 0:
        img = torchvision.utils.make_grid(data).numpy()
        # print img.shape
        # print label.shape
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        cv2.imshow('img', img)
        cv2.waitKey()
        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
