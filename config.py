import torch


class Config:
    # === 硬件 ===
    DEVICE = "cuda"
    NUM_WORKERS = 8  # CPU 拉满
    CUDNN_BENCHMARK = True

    # === 数据 ===
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_CLASSES = 10

    # === 训练策略 (高质量的核心) ===
    # 1. Batch Size: 保持大 Batch 以稳定梯度分布
    BATCH_SIZE = 176
    GRAD_ACCUM_STEPS = 2  # 显存够直接跑，不够改 2

    # 2. 学习率: 必须低！
    # 高质量生成的秘诀是“小步快跑”。1e-3 只能画轮廓，1e-4 才能画毛发。
    LR = 1e-4

    # 3. 轮数: 必须多！
    # 生成模型不像分类模型，它没有"过拟合"那么敏感。
    # 200轮是起步，SOTA 通常跑 500-1000 轮 (CIFAR-10)。
    # 5090 跑 500 轮大概只需要 3-4 小时。
    EPOCHS = 500

    # 4. EMA: 必须高！
    # 0.9999 的衰减率意味着它极度平滑，过滤掉所有训练噪声。
    EMA_DECAY = 0.9999

    # 5. 权重衰减: 关掉
    WEIGHT_DECAY = 0.0

    # === 模型结构: RSS-Flow (Base版) ===
    # 既然追求质量，就要给模型足够的"脑容量"去记忆细节。

    # RSS 模块: 增强频域处理能力
    RSS_DIM = 256  # 维度翻倍 (128 -> 256)
    MAX_RECURSION = 4  # 递归 4 次足够深了，太深边际效应递减

    # SiT 骨干: 使用 Base 规格 (参考 ViT-Base)
    SIT_HIDDEN_SIZE = 128  # 宽度翻倍 (384 -> 768)
    SIT_DEPTH = 6  # 深度保持 12
    SIT_HEADS = 2  # 768 / 64 = 12 Heads
    PATCH_SIZE = 2

    # === 日志 ===
    LOG_INTERVAL = 20
    SAMPLE_INTERVAL = 20  # 跑得久，不用看太频
    SAVE_DIR = "./checkpoints_high_quality"
    LOG_DIR = "./logs/rss_flow_high_quality"

    # 早停放宽，给它足够时间去磨细节
    PATIENCE = 50