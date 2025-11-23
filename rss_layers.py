# rss_layers.py
import torch
import torch.nn as nn
import torch.fft
from einops import rearrange
from config import Config


class SpectralGate(nn.Module):
    """
    频域混合门：在频域对幅度和相位进行全息纠缠。
    """

    def __init__(self, dim):
        super().__init__()
        # 频域特征是复数，实部+虚部使得通道数 x2
        self.complex_dim = dim * 2

        # 使用 MLP 在频域混合所有频率分量
        # 权重共享机制：这保证了它是“全息”的，不同频率共用一套物理法则
        self.mlp = nn.Sequential(
            nn.Linear(self.complex_dim, self.complex_dim),
            nn.GELU(),
            nn.Linear(self.complex_dim, self.complex_dim)
        )
        self.norm = nn.LayerNorm(self.complex_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        # 1. FFT: 空域 -> 频域
        # rfft2 输出形状: [B, C, H, W/2+1] (Complex64/128)
        x_freq = torch.fft.rfft2(x, norm='ortho')

        # 2. 整理形状: 将频率维度展平，实部虚部拼接
        x_freq_view = torch.view_as_real(x_freq)  # [B, C, H, W_freq, 2]
        # 变为 [B, H, W_freq, C*2] 以便 MLP 处理最后的一维
        x_freq_flat = rearrange(x_freq_view, 'b c h w two -> b h w (c two)')

        # 3. 频域非线性纠缠 (Residual)
        x_entangled = self.mlp(self.norm(x_freq_flat)) + x_freq_flat

        # 4. 还原形状
        x_entangled = rearrange(x_entangled, 'b h w (c two) -> b c h w two', c=C, two=2)
        x_entangled_complex = torch.view_as_complex(x_entangled.contiguous())

        # 5. iFFT: 频域 -> 空域 (此时得到的不仅是图像，而是全息干涉图)
        x_out = torch.fft.irfft2(x_entangled_complex, s=(H, W), norm='ortho')

        return x_out


class DynamicRouter(nn.Module):
    """
    动态路由：根据类别复杂度决定递归深度。
    实现方式：软门控 (Soft Gating)。
    如果是简单物体（如球），Gate值趋近0，跳过递归；
    如果是复杂物体（如森林），Gate值趋近1，执行深层递归。
    """

    def __init__(self, num_classes, dim, max_recursion):
        super().__init__()
        self.max_recursion = max_recursion

        # 类别嵌入：获取类别的"复杂度语义"
        self.class_emb = nn.Embedding(num_classes, dim)

        # 路由预测头：预测每一步递归的执行概率 (Gate)
        # 输出维度 = max_recursion，表示每一步的权重 [alpha_1, alpha_2, ...]
        self.router_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, max_recursion),
            nn.Sigmoid()  # 输出 0~1 之间的概率
        )

    def forward(self, class_labels):
        # 获取当前 batch 每个样本的递归门控值
        # gates shape: [B, max_recursion]
        emb = self.class_emb(class_labels)
        gates = self.router_head(emb)
        return gates


class RSS_Block(nn.Module):
    """
    RSS 完整模块：结合 Router 和 SpectralGate
    """

    def __init__(self, in_channels, dim, num_classes, max_recursion=3):
        super().__init__()
        self.project_in = nn.Conv2d(in_channels, dim, 1)
        self.project_out = nn.Conv2d(dim, in_channels, 1)

        self.router = DynamicRouter(num_classes, dim, max_recursion)
        self.spectral_gate = SpectralGate(dim)

        # 每次递归后的空域微调 (保持物理一致性)
        self.spatial_refine = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.SiLU()
        )

    def forward(self, x, y):
        """
        x: 输入图像/噪声 [B, C, H, W]
        y: 类别标签 [B]
        """
        # 1. 投影到特征空间
        h = self.project_in(x)
        residual = h

        # 2. 获取路由决策 [B, K]
        gates = self.router(y)

        # 3. 递归执行 (根据 Gate 软执行)
        # 理论：简单图像 Gate[k] 接近0，相当于直通；复杂图像 Gate[k] 接近1，深度纠缠
        for k in range(self.router.max_recursion):
            # 提取当前步的门控值 [B, 1, 1, 1]
            gate_k = gates[:, k].view(-1, 1, 1, 1)

            # 频域纠缠
            h_spec = self.spectral_gate(h)
            # 空域微调
            h_spec = self.spatial_refine(h_spec)

            # 软更新：New State = Gate * Entangled + (1-Gate) * Old State
            h = gate_k * h_spec + (1 - gate_k) * h

        # 4. 投影回原空间并加上残差
        out = self.project_out(h + residual)
        return out