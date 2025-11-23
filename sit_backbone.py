import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange
import warnings


# === 辅助函数 ===
def modulate(x, shift, scale):
    # x: [N, L, D], shift: [N, D], scale: [N, D]
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# === 时间步嵌入 (Sinusoidal -> MLP) ===
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


# === SiT/DiT Block (核心) ===
class SiTBlock(nn.Module):
    """
    DiT Block with adaLN-Zero conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        # Attention
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=True)

        # MLP
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim, bias=True),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size, bias=True),
        )

        # adaLN modulation
        # 输入 condition 预测 6 个参数:
        # shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        B, N, C = x.shape

        # 1. 计算调制参数
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # 2. Attention 部分
        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)

        # QKV Calculation
        qkv = self.qkv(x_norm)
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 使用 PyTorch 原生 SDPA (自动 FlashAttention)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*not compiled with flash attention.*")
            x_attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scale)

        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)

        # Residual + Gate
        x = x + gate_msa.unsqueeze(1) * x_attn

        # 3. MLP 部分
        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)
        return x


# === Final Layer (adaLN + Linear) ===
class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# === 主骨干网络 ===
class SiT_Backbone(nn.Module):
    def __init__(self,
                 img_size=32,
                 in_channels=3,
                 hidden_size=768,
                 depth=12,
                 num_heads=12,
                 patch_size=2,
                 num_classes=10):
        super().__init__()

        self.in_channels = in_channels
        self.patch_size = patch_size
        self.out_channels = in_channels

        # 1. Patch Embed
        self.x_embedder = nn.Conv2d(in_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

        # 2. Condition Embedders
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = nn.Embedding(num_classes, hidden_size)

        # 3. Positional Embedding (Learnable)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=True)

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        # 5. Final Layer
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize Patch Embeddings
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)

        # Initialize Label Embedding
        nn.init.normal_(self.y_embedder.weight, std=0.02)

        # Initialize Timestep Embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Initialize Positional Embedding
        nn.init.normal_(self.pos_embed, std=0.02)

        # Zero-out adaLN modulation layers in blocks
        # 这一步至关重要：让模型初始表现为 Identity，梯度极易传导
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out final layer
        # 这一步至关重要：让模型初始输出为 0
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y):
        """
        x: (N, C, H, W) tensor of spatial inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        # 1. Patchify & Embed
        x = self.x_embedder(x)  # (N, D, H/P, W/P)
        x = rearrange(x, 'b d h w -> b (h w) d')  # (N, L, D)
        x = x + self.pos_embed

        # 2. Add Time & Class Embeddings
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y)  # (N, D)
        c = t + y  # (N, D) - Summation is standard DiT

        # 3. Transformer Blocks
        for block in self.blocks:
            x = block(x, c)  # (N, L, D)

        # 4. Final Layer
        x = self.final_layer(x, c)  # (N, L, P*P*C)
        x = self.unpatchify(x)  # (N, C, H, W)
        return x