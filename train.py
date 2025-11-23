import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import sys
import time
import copy  # 用于 EMA
import warnings

# === 导入你的项目模块 ===
from config import Config
from dataset import get_dataloader
from full_model import RSS_Flow_Net  # 这是你的 RSS 模型
from torch.amp import GradScaler

# 屏蔽非致命警告
warnings.filterwarnings("ignore")


# ==========================================
# 1. EMA 辅助类
# ==========================================
class EMA:
    """
    指数移动平均 (Exponential Moving Average)
    用于维护一个"平滑版"的模型权重，生成质量通常远高于实时权重。
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # 深拷贝 RSS-Flow 模型
        self.ema_model = copy.deepcopy(model).eval()
        # 冻结 EMA 模型参数，不参与梯度计算
        for param in self.ema_model.parameters():
            param.requires_grad = False
        self.ema_model.to(Config.DEVICE)

    def update(self, model):
        """在 Optimizer Step 后调用"""
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), model.parameters()):
                # formula: ema = ema * decay + new * (1 - decay)
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)


# ==========================================
# 2. 采样生成函数 (适配 RSS 接口)
# ==========================================
@torch.no_grad()
def sample_images(ema_model, device, epoch, writer, num_samples=8):
    print(f"    >>> 正在使用 EMA 模型生成 Epoch {epoch} 图像...")
    ema_model.eval()

    y = torch.randint(0, Config.NUM_CLASSES, (num_samples,), device=device)
    x = torch.randn(num_samples, 3, Config.IMG_SIZE, Config.IMG_SIZE, device=device)

    # 50步 Euler 采样，保证质量
    num_steps = 50
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.ones(num_samples, device=device) * (i * dt)

        # [注意] RSS-Flow 返回 (velocity, gates)
        # 我们只需要 velocity 用于采样
        v, _ = ema_model(x, t, y)

        x = x + v * dt

    # 反归一化
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)

    grid = torchvision.utils.make_grid(x, nrow=4)
    writer.add_image('Validation/EMA_Generated', grid, epoch)


# ==========================================
# 3. 主训练循环
# ==========================================
def train():
    # 硬件优化
    torch.set_float32_matmul_precision('high')

    accum_steps = getattr(Config, 'GRAD_ACCUM_STEPS', 1)

    print(f"[{Config.DEVICE}] 启动 RSS-Flow SOTA 训练")
    print(f"配置: Batch={Config.BATCH_SIZE} | LR={Config.LR} | EMA={Config.EMA_DECAY}")
    print(f"RSS: Recursion={Config.MAX_RECURSION} | Dim={Config.RSS_DIM}")

    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(Config.LOG_DIR)

    train_loader = get_dataloader(train=True)

    # 1. 初始化主模型
    model = RSS_Flow_Net().to(Config.DEVICE)

    # 2. 初始化 EMA 模型
    ema = EMA(model, decay=Config.EMA_DECAY)

    # 3. 优化器配置 (AdamW, 1e-4, No WD)
    # 虽然 Config.WEIGHT_DECAY 已经是 0，但我们保留分组逻辑以防万一
    # 这样可以确保 bias, norm, rss_out 绝对不被衰减
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2 and 'project_out' not in n]
    no_decay_params = [p for n, p in param_dict.items() if p.dim() < 2 or 'project_out' in n]

    optim_groups = [
        {'params': decay_params, 'weight_decay': Config.WEIGHT_DECAY},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    optimizer = torch.optim.AdamW(optim_groups, lr=Config.LR)
    scaler = GradScaler('cuda')

    # 4. 训练状态
    global_step = 0
    start_time = time.time()

    # 清空梯度
    optimizer.zero_grad(set_to_none=True)

    try:
        for epoch in range(Config.EPOCHS):
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", file=sys.stdout)

            running_loss = 0.0

            for i, (x1, labels) in enumerate(progress_bar):
                x1 = x1.to(Config.DEVICE, non_blocking=True)
                labels = labels.to(Config.DEVICE, non_blocking=True)
                B = x1.shape[0]

                # Flow Matching
                x0 = torch.randn_like(x1)
                t = torch.rand(B, device=Config.DEVICE)
                t_view = t.view(B, 1, 1, 1)
                x_t = t_view * x1 + (1 - t_view) * x0
                target_v = x1 - x0

                # Forward (BF16)
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    # RSS-Flow 返回 v 和 gates
                    pred_v, gates = model(x_t, t, labels)
                    loss = F.mse_loss(pred_v, target_v)
                    loss = loss / accum_steps

                # Backward
                scaler.scale(loss).backward()

                # Update
                if (i + 1) % accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    # [关键] 更新 EMA
                    ema.update(model)

                    global_step += 1

                    # Logging
                    if global_step % Config.LOG_INTERVAL == 0:
                        loss_val = loss.item() * accum_steps
                        writer.add_scalar('Loss/train', loss_val, global_step)

                        # 监控 RSS 递归深度 (平均 Gate 值)
                        # sum(gates, dim=1) 是每个样本的总递归权重
                        avg_depth = gates.detach().sum(dim=1).mean().item()
                        writer.add_scalar('RSS/Avg_Recursion_Depth', avg_depth, global_step)

                        # 监控 RSS 是否苏醒 (Output Weight Norm)
                        # 如果这个值从 0 变大，说明 RSS 成功介入
                        if hasattr(model, 'rss_module'):
                            out_w_norm = model.rss_module.project_out.weight.data.norm().item()
                            writer.add_scalar('RSS_Debug/Out_W_Norm', out_w_norm, global_step)

                # 进度条显示 Loss
                current_loss = loss.item() * accum_steps
                running_loss += current_loss
                progress_bar.set_postfix(loss=f"{current_loss:.4f}")

            # Epoch 结束
            epoch_loss = running_loss / len(train_loader)
            print(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f}")

            # 采样与保存
            if (epoch + 1) % Config.SAMPLE_INTERVAL == 0:
                # 使用 EMA 模型采样
                sample_images(ema.ema_model, Config.DEVICE, epoch + 1, writer)

                # 保存 Checkpoint
                save_path = os.path.join(Config.SAVE_DIR, "latest.pth")
                torch.save({
                    'model': model.state_dict(),
                    'ema': ema.ema_model.state_dict(),  # 保存 EMA 权重
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                }, save_path)
                print(f"Checkpoint saved to {save_path}")

    except KeyboardInterrupt:
        print("\n>>> Training Interrupted. Saving EMA state...")
        save_path = os.path.join(Config.SAVE_DIR, "interrupted.pth")
        torch.save({
            'model': model.state_dict(),
            'ema': ema.ema_model.state_dict(),
        }, save_path)
        print("Done.")

    writer.close()


if __name__ == "__main__":
    train()