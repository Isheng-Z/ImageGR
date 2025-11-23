import torch
import torch.nn as nn
import torch.fft
from einops import rearrange
from config import Config


class SpectralGate(nn.Module):
    """
    [修改版] 渐进式频域混合门
    不再一次性剧烈打乱，而是引入可学习的 scale 因子。
    """

    def __init__(self, dim):
        super().__init__()
        self.complex_dim = dim * 2

        self.mlp = nn.Sequential(
            nn.Linear(self.complex_dim, self.complex_dim),
            nn.GELU(),
            nn.Linear(self.complex_dim, self.complex_dim)
        )
        self.norm = nn.LayerNorm(self.complex_dim)

        # [新增] 可学习的缩放因子，初始化为很小的值 (0.1)
        # 这保证了初始时每次递归只引入微小的"混沌"
        self.scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # 强制 FP32 以保证 FFT 精度
        with torch.amp.autocast('cuda', enabled=False):
            x = x.float()
            B, C, H, W = x.shape

            # FFT
            x_freq = torch.fft.rfft2(x, norm='ortho')

            # Reshape
            x_freq_view = torch.view_as_real(x_freq)
            x_freq_flat = rearrange(x_freq_view, 'b c h w two -> b h w (c two)')

            # [核心修改] 渐进式混合
            # out = x + scale * MLP(x)
            # 这样保证了原始频率信息占主导，MLP 只是添加微扰
            perturbation = self.mlp(self.norm(x_freq_flat))
            x_mixed = x_freq_flat + self.scale * perturbation

            # Reshape back
            x_mixed = rearrange(x_mixed, 'b h w (c two) -> b c h w two', c=C, two=2)
            x_mixed_complex = torch.view_as_complex(x_mixed.contiguous())

            # iFFT
            x_out = torch.fft.irfft2(x_mixed_complex, s=(H, W), norm='ortho')

        return x_out


class DynamicRouter(nn.Module):
    def __init__(self, num_classes, dim, max_recursion):
        super().__init__()
        self.max_recursion = max_recursion
        self.class_emb = nn.Embedding(num_classes, dim)

        # 预测每一步的概率
        self.router_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, max_recursion),
            nn.Sigmoid()
        )

    def forward(self, class_labels):
        emb = self.class_emb(class_labels)
        gates = self.router_head(emb)
        return gates


class RSS_Block(nn.Module):
    def __init__(self, in_channels, dim, num_classes, max_recursion):
        super().__init__()
        self.project_in = nn.Conv2d(in_channels, dim, 1)
        # 零初始化输出层 (保证初始不干扰 SiT)
        self.project_out = nn.Conv2d(dim, in_channels, 1)
        nn.init.zeros_(self.project_out.weight)
        nn.init.zeros_(self.project_out.bias)

        self.router = DynamicRouter(num_classes, dim, max_recursion)
        self.spectral_gate = SpectralGate(dim)

        self.spatial_refine = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.SiLU()
        )

    def forward(self, x, y):
        h = self.project_in(x)
        residual = h

        # 获取门控值 [B, Max_Recursion]
        gates = self.router(y)

        for k in range(self.router.max_recursion):
            gate_k = gates[:, k].view(-1, 1, 1, 1)

            # 频域微扰
            h_spec = self.spectral_gate(h)
            # 空域整理
            h_spec = self.spatial_refine(h_spec)

            # 软更新: 只有当 Gate 开启时，才累加这次的扰动
            h = gate_k * h_spec + (1 - gate_k) * h

        out = self.project_out(h + residual)
        return out, gates  # [修改] 返回 gates 以便 Tensorboard 监控