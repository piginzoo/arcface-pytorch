import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

logger = logging.getLogger(__name__)


class ArcMarginProduct(nn.Module):
    """Implement of large margin arc distance: :
        Args:
            in_features: size of each input sample
            out_features: size of each output sample
            s: norm of input feature，其实我理解就是余弦夹角的半径长度了，可以形象的理解为
            m: margin 是一个角度的margin，你可以理解是一段弧长
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False, device='cuda'):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features  # 入参是人脸向量，512，定死了
        self.out_features = out_features  # 这个就是人脸分类，1万多，就是不同人
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)  # 初始化

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.device = device

    def forward(self, input, label):
        """
        @param input: 512维向量
        @param label:

        其实就是在实现 softmax中的子项 exp( s * cos(θ + m) )，
        但是因为cos里面是个和：θ + m
        所以要和差化积，就得分解成：
        - exp( s * cos(θ + m) )
        - cos(θ + m) = cos(θ) * cos(m) - sin(θ) * sin(m) = cos_θ_m(程序中的中间变量) # 和差化积
        - sin(θ) = sqrt( 1 - cos(θ)^2 )
        - cos(θ) = X*W/|X|*|W|
        s和m是超参： s - 分类的半径；m - 惩罚因子

        这个module得到了啥？得到了一个可以做softmax的时候，归一化的余弦最大化的向量
        """

        logger.debug("[网络输出]arcface的loss的输入x：%r, label: %r", input.shape,label)
        # --------------------------- cos(θ) & phi(θ) ---------------------------
        """
        >>> F.normalize(torch.Tensor([[1,1],
                                      [2,2]]))
            tensor([[0.7071, 0.7071],
                    [0.7071, 0.7071]])
        这里有点晕，需要解释一下，cosθ = x.W/|x|*|W|, 
        注意，x.W表示点乘，而|x|*|W|是一个标量间的相乘，所以cosθ是一个数（标量）
        可是，你如果看下面这个式子`cosine = F.linear(F.normalize(input), F.normalize(self.weight))`，
        你会发现，其结果是10000（人脸类别数），为何呢？cosθ不应该是个标量？为何现在成了10000的矢量了呢？
        思考后，我终于理解了，注意，这里的x是小写，而W是大写的，这个细节很重要，
        x是[Batch,512]，而W是[512,10000]，
        而其实，我们真正要算的是一个512维度的x和一个10000维度的W_i，他们cosθ = x.W_i/|x|*|W_i|，这个确实是一个标量。
        但是，我们有10000个这样的W_i，所以，我们确实得到了10000个这样的cosθ，明白了把！
        所以，这个代码就是实现了这个逻辑。没问题。
        
        再多说一句，arcface，就是要算出10000个θ，这1万个θ，接下来
        """
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))  # |x| * |w|
        # logger.debug("[网络输出]cos：%r", cosine.shape)

        # clamp，min~max间，都夹到范围内 : https://blog.csdn.net/weixin_40522801/article/details/107904282
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0,1))
        # logger.debug("[网络输出]sin：%r", sine.shape)

        # 和差化积，cos(θ + m) = cos(θ) * cos(m) - sin(θ) * sin(m)
        cos_θ_m = cosine * self.cos_m - sine * self.sin_m

        logger.debug("[网络输出]cos_θ_m：%r", cos_θ_m.shape)
        if self.easy_margin:
            cos_θ_m = torch.where(cosine > 0, cos_θ_m, cosine)
        else:
            # th = cos(π - m) ，mm = sin(π - m) * m
            # ？？？为何要做这个？？？
            cos_θ_m = torch.where(cosine > self.th, cos_θ_m, cosine - self.mm)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device=self.device)
        # logger.debug("[网络输出]one_hot：%r", one_hot.shape)

        # input.scatter_(dim, index, src)：从【src源数据】中获取的数据，按照【dim指定的维度】和【index指定的位置】，替换input中的数据。
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        # 这步是在干嘛？是在算arcloss损失函数（论文2.1节的L3）的分母，
        # 标签对的那个分类y_i项是s*cos(θ_yi + m)，而其他分类则为s*cos(θ_yj), 其中j!=i，
        # 所以这个'骚操作'是为了干这件事：
        output = (one_hot * cos_θ_m) + ((1.0 - one_hot) * cosine)

        # logger.debug("[网络输出]output：%r", output.shape)
        output *= self.s

        # logger.debug("[网络输出]arcface的loss最终结果：%r", output.shape)
        # 输出是啥？？？ => torch.Size([10, 10178]
        # 自问自答：输出是softmax之前的那个向量，注意，softmax只是个放大器，
        # 我们就是在准备这个放大器的输入的那个向量，是10178维度的，[cosθ_0,cosθ_1,...,cos(θ_{i-1}),cos(θ_i+m),cos(θ_{i+1}),...]
        #                                           只有这项是特殊的,θ_i多加了m，其他都没有---> ~~~~~~~~~~
        # 不是概率，概率是softmax之后才是概率
        return output


class AddMarginProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(θ) - m
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.40):
        super(AddMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label):
        # --------------------------- cos(θ) & phi(θ) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        phi = cosine - self.m
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cosine.size(), device='cuda')
        # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
                (1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', s=' + str(self.s) \
               + ', m=' + str(self.m) + ')'


class SphereProduct(nn.Module):
    r"""Implement of large margin cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        m: margin
        cos(m*theta)
    """

    def __init__(self, in_features, out_features, m=4):
        super(SphereProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.base = 1000.0
        self.gamma = 0.12
        self.power = 1
        self.LambdaMin = 5.0
        self.iter = 0
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform(self.weight)

        # duplication formula
        self.mlambda = [
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]

    def forward(self, input, label):
        # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
        self.iter += 1
        self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

        # --------------------------- cos(θ) & phi(θ) ---------------------------
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m_theta = self.mlambda[self.m](cos_theta)
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.14159265).floor()
        phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
        NormOfFeature = torch.norm(input, 2, 1)

        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros(cos_theta.size())
        one_hot = one_hot.cuda() if cos_theta.is_cuda else one_hot
        one_hot.scatter_(1, label.view(-1, 1), 1)

        # --------------------------- Calculate output ---------------------------
        output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
        output *= NormOfFeature.view(-1, 1)

        return output

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'in_features=' + str(self.in_features) \
               + ', out_features=' + str(self.out_features) \
               + ', m=' + str(self.m) + ')'
