# 人脸比较-arcface

# 数据

旧代码用的训练集是Webface数据：[CASIA-webface](https://paperswithcode.com/dataset/casia-webface)做训练集，
我改成了CelebA数据集作为训练集：[CelebA数据集](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
测试集仍然使用原代码的lfw数据集，用于计算acc：[LfW数据集](http://home.ustc.edu.cn/~yqli1995/2019/09/28/lfw/)

```
训练数据
train    <--- CelebA数据集
├── Anno
│   └── identity_CelebA.txt
└── Img
    └── img_align_celeba
        ├── 010353.jpg
        ├── 134059.jpg
        └── 139608.jpg

测试数据
val     <--- lfw数据集
├── images
│        ├── AJ_Cook
│        │       └── AJ_Cook_0001.jpg
│        ├── AJ_Lamas
│        │       └── AJ_Lamas_0001.jpg
│        ├── Aaron_Eckhart
│        │       └── Aaron_Eckhart_0001.jpg
```

# 原文档
```
# arcface-pytorch
pytorch implement of arcface 

# References
https://github.com/deepinsight/insightface

https://github.com/auroua/InsightFace_TF

https://github.com/MuggleWang/CosFace_pytorch

# pretrained model and lfw test dataset
the pretrained model and the lfw test dataset can be download here. link: https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA pwd: b2ec
the pretrained model use resnet-18 without se. Please modify the path of the lfw dataset in config.py before you run test.py.
```