# 人脸比较-arcface

# 数据
旧代码用的是
用[CASIA-webface](https://paperswithcode.com/dataset/casia-webface)做训练集，
用lfw数据集做验证集，算acc[参考](http://home.ustc.edu.cn/~yqli1995/2019/09/28/lfw/)
```
➜  lfw ll images/Aaron_Peirsol
total 136
-rw-r--r--  1 piginzoo  staff    13K 10  7  2007 Aaron_Peirsol_0001.jpg
-rw-r--r--  1 piginzoo  staff    16K 10  7  2007 Aaron_Peirsol_0002.jpg
-rw-r--r--  1 piginzoo  staff    14K 10  7  2007 Aaron_Peirsol_0003.jpg
-rw-r--r--  1 piginzoo  staff    16K 10  7  2007 Aaron_Peirsol_0004.jpg

# 正例
Aaron_Peirsol    1    2
Aaron_Peirsol    3    4
# 负例
AJ_Cook    1    Marsha_Thomason    1
Aaron_Sorkin    2    Frank_Solich    5

1，2，3，4，5是图片的编号：Aaron_Peirsol_0003.jpg => 3
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