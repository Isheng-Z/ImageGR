import torch
import torch.nn as nn
from config import Config
from sit_backbone import SiT_Backbone


class Pure_SiT_Net(nn.Module):
    """
    基线模型：没有 RSS，没有递归，只有标准的 SiT。
    用于验证 Flow Matching Pipeline 是否正常。
    """

    def __init__(self):
        super().__init__()

        # 直接实例化 SiT 骨干
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
        x: Noisy Image [B, C, H, W]
        t: Time [B]
        y: Label [B]
        """
        # 直接进入 Transformer，没有任何频域预处理
        velocity = self.sit_module(x, t, y)

        return velocity