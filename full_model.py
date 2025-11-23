# full_model.py
import torch.nn as nn
from config import Config
from rss_layers import RSS_Block
from sit_backbone import SiT_Backbone


class RSS_Flow_Net(nn.Module):
    """
    RSS-Flow 完整架构:
    Step 1: RSS (Chaos Generator) - 制造全息纠缠态
    Step 2: SiT (Manifold Projector) - 投影回图像流形
    """

    def __init__(self):
        super().__init__()

        # 前半段：递归频域模块
        self.rss_module = RSS_Block(
            in_channels=Config.IN_CHANNELS,
            dim=Config.RSS_DIM,
            num_classes=Config.NUM_CLASSES,
            max_recursion=Config.MAX_RECURSION
        )

        # 后半段：SiT 骨干
        self.sit_module = SiT_Backbone(
            img_size=Config.IMG_SIZE,
            in_channels=Config.IN_CHANNELS,
            hidden_size=Config.SIT_HIDDEN_SIZE,
            depth=Config.SIT_DEPTH,
            num_heads=Config.SIT_HEADS,
            patch_size=Config.PATCH_SIZE,
            num_classes=Config.NUM_CLASSES
        )

    def forward(self, x, t, y):
        """
        x: 当前状态 (噪声 or 半成品图像)
        t: 时间步
        y: 类别标签
        """
        # 1. 频域预处理：增加逻辑深度
        # 注意：我们把 RSS 的输出作为一种“增强特征”或者“变换后的坐标”
        # 为了防止梯度消失，采用残差连接：RSS(x) + x
        rss_feat = self.rss_module(x, y)

        # 融合策略：将 RSS 的混沌态叠加到输入上
        sit_input = x + rss_feat

        # 2. 流形投影：预测速度场 v
        velocity = self.sit_module(sit_input, t, y)

        return velocity