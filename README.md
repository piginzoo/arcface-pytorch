# 人脸比较-arcface

关于原理和代码，可以参考[我的博客](http://www.piginzoo.com/machine-learning/2021/09/15/face-recognition#arcface)。

# 如何运行

1、构建docker容器：`bin/build.sh`

2、训练：
   - 训练mnist数据集，纯softmax：`bin/train.docker 1 mnist.ce`, 1是显卡的序号，从0开始
   - 训练mnist数据集，使用arcface：`bin/train.docker 1 mnist.arcface`
   - 训练人脸数据集，使用arcface：`bin/train.docker 1 face`

3、单独跑测试：`bin/train.docker 1 test`，这个集成到训练过程中了，也可以单独跑

4、启动tensorboard：`bin/tboard.sh`，可以观察2维可视化、查看训练中的样本图像采样、查看loss/acc等指标

# 关于训练

我们使用了CeleBa的训练集（20万张），LFW的验证集（6000张）。

由于我们服务器上是CUDA7，对应pytorch是1.0，程序有很多问题。
于是，我们使用了docker方式训练（docker可以封装CUDA，且像显卡驱动低版本兼容），
我们使用了tensorflow2.0-gpu的容器，然后跑了一个[Dockerfile](deploy/Dockerfile)，
build出来我们自己的训练容器：[arcface.img:v1]。
然后使用命令[train.docker](bin/train.docker)，会启动容器进行训练。

训练不收敛，后来，尝试了Mnist数据集，用一个小的数据集确认了arcface的有效性，
虽然可视化感觉是收敛的（2维可视化），但是和论文里那种聚簇到一起、类间距很大的效果还是差很多。

最终的结果：

在最后的各类调整后，结果收敛：
- 用了resnet的avg层，放弃了5x5x2048的方式，减少了全连接的参数量
- m=1.2，s=64
- 设置easy_margin = True，(夹角超过180度，CosFace 代替 ArcFace)

训练大概10个小时，1万个分类，train acc接近到1，test acc大概是82%，还和论文有一定差距，需要再优化。

训练的详细过程，防止在[训练日志](doc/developing%20logs.docx)中，以供参考

# Fork改进
基于原有版本，我们做了如下改进：
- 追加注释，对每行代码理解，与论文严格对应并注释
- 构建了上述的训练docker容器，方便pytorch1.7-gpu版本的docker容器方式训练
- 实现了一个ealy stop，来方便早停
- 改造了Visualizer，方便与容器外的visdom服务器通讯
- 还增加了tensorboard的支持，作为visdom的备份（visdom不好用）
- 改造了数据集加载，配合使用下面说的数据集加载
- 为了对比验证，增加了Mnist数据集的测试，分别测试一般的全连接，和arcface的约束
- 增加了mnist的可视化plot，画了一个二维的环形图，用于和论文的效果做对比

# 相关数据

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

- 网络的改进
    为了验证arcface可以类内聚，类间分开，我构建了一个mnist网络，用来通过可视化plot来展示分离效果，
    trick是要把embedding搞成2维，这样也不需要用什么tsne/pca了，但是2维肯定无法表示人脸，
    所以才引入mnist数据集来测试的，我看论文中也都是使用mnist和二维embedding来验证loss函数的有效性的。
    
    人脸的网络，其实是用resnet做了特征抽取后，再接了一个全连接降维到128，然后在用128和1万个分类做全连接，
    之后，再加上和Weight（也就是类中心）的约束后，得到当前embedding和每个类的cos值，用cos值再做交叉熵。
    
    参数量：
    ```
    resnet50：23454912
    降维FC：     262144 = 2048x128
    分类FC：    1280000 = 128x10000
    合计：     24997056 = 2500万
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