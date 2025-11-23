# config.py
import torch


class Config:
    # 硬件与环境
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 2  # Windows下建议设为0-4，设太大可能会报错

    # 数据集
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_CLASSES = 10
    BATCH_SIZE = 64

    # RSS (前端频域模块) 参数
    RSS_DIM = 128  # RSS模块的隐藏层维度
    MAX_RECURSION = 3  # 路由器的最大递归深度 (Max K)

    # SiT (后端流形投影) 参数
    SIT_HIDDEN_SIZE = 384  # SiT-Small 规模
    SIT_DEPTH = 12  # Transformer 层数
    SIT_HEADS = 6  # 注意力头数
    PATCH_SIZE = 2  # CIFAR-10 图像小，Patch要小

    # 训练参数
    LR = 3e-4
    EPOCHS = 100
    LOG_INTERVAL = 50  # 多少个batch打印一次日志
    SAVE_DIR = "./checkpoints"
    LOG_DIR = "./logs/rss_flow_step2"