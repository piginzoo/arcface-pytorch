"""
用来测试图片是不是transform得合适
"""

from PIL import Image
from torchvision import transforms as T  # PyTorch框架中有一个非常重要且好用的包

dir = "/Users/piginzoo/Downloads/train_images/人脸/CelebA/Img/img_align_celeba"
dst = "data/temp"
import os

if not os.path.exists(dst):
    os.makedirs(dst)

size = (160, 160)

for f in os.listdir(dir):
    path = os.path.join(dir, f)
    data = Image.open(path)  # 加载图片
    data = data.convert('L')  # a greyscale ("L") ，L是灰度图像
    # image = T.RandomCrop()(data)

    transforms = T.Compose([  # 做增强
        T.RandomCrop(size),  #
        T.RandomHorizontalFlip()  # 把图片水平方向翻过来，有点像镜子里看
    ])
    image = transforms(data)

    image.save(os.path.join(dst, f))

# python -m sandbox.test_transforms
