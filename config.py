import torch


class Config:
    # 硬件
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_WORKERS = 4
    CUDNN_BENCHMARK = True

    # 数据
    IMG_SIZE = 32
    IN_CHANNELS = 3
    NUM_CLASSES = 10
    BATCH_SIZE = 256  # 保持这个，显存够用

    # === RSS 瘦身 ===
    RSS_DIM = 64  # 原 128 -> 降为 64
    MAX_RECURSION = 4  # 原 8 -> 降为 2 (验证理论只需有递归即可，不需要太深)

    # === SiT 瘦身 (SiT-Nano) ===
    SIT_HIDDEN_SIZE = 128 # 原 384 -> 降为 128
    SIT_DEPTH = 6  # 原 12 -> 降为 6
    SIT_HEADS = 4 # 原 6 -> 降为 4
    PATCH_SIZE = 2

    # 训练
    LR =1e-3  # 模型变小，LR 可以大一点
    EPOCHS = 50  # MVP 跑 50 轮足够看清效果了
    LOG_INTERVAL = 10
    SAMPLE_INTERVAL = 5  # 每5轮看一次图

    # 路径
    SAVE_DIR = "./checkpoints"
    LOG_DIR = "./logs/rss_flow_mvp_fast"

    # === 早停策略 ===
    PATIENCE = 5  # 容忍多少个 Epoch Loss 不下降
    MIN_DELTA = 0.001  # Loss 至少要下降多少才算有效