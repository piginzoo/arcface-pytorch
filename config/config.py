import sys


class Config(object):
    env = 'default'
    backbone = 'resnet50'
    classify = 'softmax'
    num_classes = 10177  # webface数据集10575,celeba数据集10177个分类
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = True
    finetune = False

    train_root = 'data/train/Img/img_align_celeba/'
    train_list = 'data/train/Anno/identity_CelebA.txt'

    train_root = 'data/train/Img/img_align_celeba/'
    train_list = 'data/train/Anno/identity_CelebA.txt'

    checkpoints_path = 'checkpoints/arcface_e{}_s{}_{}_l{:.2f}_a{:.2f}.model'  # epoch,step,datetime,loss,acc

    train_batch_size = 16  # batch size

    lfw_root = 'data/val/images'
    lfw_test_pair_path = 'data/val/lfw_test_pair.txt'
    test_model_path = 'checkpoints/resnet18_110.pth'
    test_batch_size = 60
    test_size = sys.maxsize

    early_stop = 100

    # 对celeba数据集原图是178x218=>(170,170)，对lfw和webface原图是250x250=>(240,240)，
    # 原来程序给的128x128肯定是不合适的，会切丢的，他切是按照原尺寸和目标尺寸之间的间隙随机动，所以原图和目标size相差不能太大（参加RandomCrop源码）
    input_shape = (3, 160, 160)

    use_gpu = False  # use GPU or not
    gpu_id = '0, 1'
    num_workers = 4  # how many workers for loading data
    print_freq = 1000  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4

    # 日志信息
    tensorboard_dir = "logs/tboard"
    visdom_port = 8086
    visualizer = "tensorboard"
