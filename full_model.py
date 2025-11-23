import torch.nn as nn
from config import Config
from rss_layers import RSS_Block
from sit_backbone import SiT_Backbone


class RSS_Flow_Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.rss_module = RSS_Block(
            in_channels=Config.IN_CHANNELS,
            dim=Config.RSS_DIM,
            num_classes=Config.NUM_CLASSES,
            max_recursion=Config.MAX_RECURSION
        )

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
        # 1. 频域递归处理 (带有微扰动和路由器)
        # gates 用于监控递归深度
        rss_feat, gates = self.rss_module(x, y)

        # 2. 叠加特征
        sit_input = x + rss_feat

        # 3. 流形投影
        velocity = self.sit_module(sit_input, t, y)

        return velocity, gates