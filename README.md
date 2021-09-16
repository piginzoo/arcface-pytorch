# 人脸比较-arcface

# 训练

我们使用了CeleBa的训练集（20万张），LFW的验证集（6000张）。

由于我们服务器上是CUDA7，对应pytorch是1.0，程序有很多问题。
于是，我们使用了docker方式训练（docker可以封装CUDA，且像显卡驱动低版本兼容），
我们使用了tensorflow2.0-gpu的容器，然后跑了一个[Dockerfile](deploy/Dockerfile)，
build出来我们自己的训练容器：[arcface.img:v1]。
然后使用命令[train.docker](bin/train.docker)，会启动容器进行训练。

# 改进
基于原有版本，我们做了如下改进：
- 追加注释，对每行代码理解，与论文严格对应并注释
- 构建了上述的训练容器，方便pytorch1.7-gpu版本的训练
- 实现了一个ealystop，来方便早停
- 改造了Visualizer，方便与容器外的visdom服务器通讯
- 还增加了tensorboard的支持，作为visdom的备份（visdom不好用）
- 改造了数据集加载，配合使用下面说的数据集加载

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

# 一些细节

- 构建容器
    容器其实不太好构建，你得清楚docker容器是包含CUDA的，且，CUDA是向下兼容显卡driver的，否则，你可能会逼着自己把服务器显卡驱动升级，
    我不是说升级不好，但是升级就需要重启服务器，我不想这么折腾。
    
- 代码理解
    loss里面的exp( s * cos(θ + m) )，需要使用三角函数的和差化积，你要是不理解这个细节，就会盯着代码发蒙半天
    另外，就是要对arcface的loss深刻理解，特别是理解W就是分类的中心，如果你不知道我在说什么，请回去看论文
    
- 数据集
    原作者使用的是webface数据集，10575个人，494414张人脸图像（1万人，50万张脸），不过我看有人说，数据有噪音，
    所以，我就选择了CeleBrayA数据集，10177人，202599万张图片，说是噪音少一些，好吧，其实对数据集的选择很主观的。
    所以，用lfw，另外一个数据集来做test，也算是公平把。
    未来，我可以再用webface训练一遍，看看到底效果是否可以提升吧。
    ```
        wc -l identity_CelebA.txt
        > 202599 identity_CelebA.txt <---- 图像确实202599张
        
        cat identity_CelebA.txt|awk '{print $2}'|sort|uniq|wc -l
        >10177 <---- 分类数确实是10177
    ```
    
    常见的数据集：
    
    **人脸识别数据集**
    
    ```
        WebFace	    10k+人，约500K张图片	非限制场景，图像250x250
        FaceScrub	530人，约100k张图片	非限制场景
        YouTubeFace	1,595个人 3,425段视频	非限制场景、视频
        LFW	        5k+人脸，超过10K张图片	标准的人脸识别数据集，图像250x250
        MultiPIE	337个人的不同姿态、表情、光照的人脸图像，共750k+人脸图像	限制场景人脸识别	
        MegaFace	690k不同的人的1000k人脸图像	新的人脸识别评测集合
        IJB-A	 	人脸识别，人脸检测
        CAS-PEAL	1040个人的30k+张人脸图像，主要包含姿态、表情、光照变化	限制场景下人脸识别
        Pubfig	    200个人的58k+人脸图像	非限制场景下的人脸识别
        CeleBrayA	200k张人脸图像40多种人脸属性，图像178x218
    ```

    **人脸检测数据集**
    ```
    FDDB	            2845张图片中的5171张脸	标准人脸检测评测集	链接
    IJB-A	 	        人脸识别，人脸检测	链接
    Caltech10k WebFaces	10k+人脸，提供双眼和嘴巴的坐标位置	人脸点检测	链接
    ```
- 图像加载
    训练图像用的是PLI加载后，调用pytorch的transforms包，来的调用的，这个包必须要求是PIL的image。
    而测试的图像加载使用的是cv2，为了保持一致，需要把cv2的GBR通道转化成RGB通道，
    另外，训练的图像都做了归一化，所以，测试加载也把图像做了归一化处理（ - 127.5 / 255 )
    注意，这个不是严格意义上的归一化，应该是 - mean/max，不过，训练也是这样做的，所以就OK了。

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