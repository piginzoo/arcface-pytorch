import cv2
import numpy as np
from PIL import Image

image_path = "data/train/Img/img_align_celeba/010353.jpg"

# 测试1：用PIL加载，得到RGB数组，但是cv2当做GBR去保存
image = Image.open(image_path)
image_numpy = np.asarray(image)
image_numpy = image_numpy[:, :, ::-1]
cv2.imwrite("data/temp/test.1.jpg", image_numpy)
# 测试结果，果然是这种转化得到的GBR的图像，也就是说，cv2认为图像格式就是GBR的，所以不得不做一次:,:,::-1

# 测试2：用cv2加载，用PIL保存
image_numpy = cv2.imread(image_path)
image_numpy = image_numpy[:, :, ::-1]
image = Image.fromarray(image_numpy)
image.save("data/temp/test.2.jpg")
# 测试结果，cv2加载出来是的GBR，所以不得不做一次:,:,::-1，变成RGB，再用PIL.Image去保存

# python -m sandbox.test_image_load && see data/temp/test.*
