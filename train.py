import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
import sys
import time
import warnings

# 项目依赖
from config import Config
from dataset import get_dataloader
from full_model import RSS_Flow_Net
from torch.amp import GradScaler

# 屏蔽非致命警告
warnings.filterwarnings("ignore")


# === 早停辅助类 ===
class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


# === 采样生成函数 ===
@torch.no_grad()
def sample_images(model, device, epoch, writer, num_samples=8):
    model.eval()
    # 随机采样类别和噪声
    y = torch.randint(0, Config.NUM_CLASSES, (num_samples,), device=device)
    x = torch.randn(num_samples, 3, Config.IMG_SIZE, Config.IMG_SIZE, device=device)

    # Euler 采样步数 (验证时20步足够看清轮廓)
    num_steps = 20
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.ones(num_samples, device=device) * (i * dt)
        # 注意：这里接收两个返回值，但采样时只需要 velocity
        v, _ = model(x, t, y)
        x = x + v * dt

    # 反归一化并记录
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    grid = torchvision.utils.make_grid(x, nrow=4)
    writer.add_image('Validation/Generated_Images', grid, epoch)
    model.train()


# === 主训练函数 ===
def train():
    # 硬件优化
    torch.set_float32_matmul_precision('high')
    if Config.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    print(f"[{Config.DEVICE}] 训练启动 (监控增强版)")
    print(f"配置: Max Recursion: {Config.MAX_RECURSION} | Batch: {Config.BATCH_SIZE} | BF16: On")
    print("提示: 按 Ctrl+C 可保存当前进度并退出。")

    # 初始化
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    writer = SummaryWriter(Config.LOG_DIR)

    train_loader = get_dataloader(train=True)
    model = RSS_Flow_Net().to(Config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = GradScaler('cuda')
    early_stopper = EarlyStopper(patience=Config.PATIENCE if hasattr(Config, 'PATIENCE') else 5, min_delta=0.001)

    global_step = 0
    start_time = time.time()

    try:
        for epoch in range(Config.EPOCHS):
            model.train()
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}", file=sys.stdout)

            running_loss = 0.0
            avg_recursion_depth = 0.0

            for x1, labels in progress_bar:
                x1 = x1.to(Config.DEVICE, non_blocking=True)
                labels = labels.to(Config.DEVICE, non_blocking=True)
                B = x1.shape[0]

                # Flow Matching 数据构造
                x0 = torch.randn_like(x1)
                t = torch.rand(B, device=Config.DEVICE)
                t_view = t.view(B, 1, 1, 1)
                x_t = t_view * x1 + (1 - t_view) * x0
                target_v = x1 - x0

                optimizer.zero_grad(set_to_none=True)

                # 前向传播 (BF16)
                with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                    # 获取预测速度 v 和 路由门控 gates
                    pred_v, gates = model(x_t, t, labels)
                    loss = F.mse_loss(pred_v, target_v)

                # 反向传播
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # === [监控] RSS 内部梯度 (在 Clip 之前记录) ===
                if global_step % Config.LOG_INTERVAL == 0:
                    if hasattr(model, 'rss_module') and model.rss_module.project_in.weight.grad is not None:
                        grad_norm = model.rss_module.project_in.weight.grad.norm().item()
                        writer.add_scalar('RSS_Debug/Input_Grad_Norm', grad_norm, global_step)

                # 梯度裁剪与更新
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                # scheduler.step()

                # 统计数据
                loss_val = loss.item()
                running_loss += loss_val

                # 计算当前 batch 的平均递归深度
                curr_depth = gates.detach().sum(dim=1).mean().item()
                avg_recursion_depth += curr_depth

                # === TensorBoard 详细记录 ===
                if global_step % Config.LOG_INTERVAL == 0:
                    # 1. 基础指标
                    writer.add_scalar('Loss/train', loss_val, global_step)
                    writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

                    # 2. 速度指标
                    elapsed = time.time() - start_time
                    img_per_sec = (global_step * B) / (elapsed + 1e-6)
                    writer.add_scalar('Speed/img_per_sec', img_per_sec, global_step)

                    # 3. RSS 核心指标 (验证理论的关键)
                    writer.add_scalar('RSS/Avg_Recursion_Depth', curr_depth, global_step)

                    # 4. RSS 零初始化唤醒监控
                    # 监控 Output Project 层权重的 L2 Norm
                    # 如果这个值从 0 开始上升，说明 RSS 模块被"激活"了
                    out_weight_norm = model.rss_module.project_out.weight.data.norm().item()
                    writer.add_scalar('RSS_Debug/Output_Weight_Norm', out_weight_norm, global_step)

                    # 5. Gate 分布监控 (取第一个样本的第一层 Gate)
                    first_gate = gates[0, 0].item()
                    writer.add_scalar('RSS_Debug/Gate_Layer0_Sample', first_gate, global_step)

                global_step += 1

            # Epoch 结束处理
            epoch_loss = running_loss / len(train_loader)
            epoch_depth = avg_recursion_depth / len(train_loader)
            print(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f} | Avg Depth: {epoch_depth:.2f}")

            # 保存最新权重
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "latest.pth"))

            # 采样生成
            if (epoch + 1) % Config.SAMPLE_INTERVAL == 0:
                print(">>> 正在生成验证图片...")
                sample_images(model, Config.DEVICE, epoch + 1, writer)

            # 早停检查
            if early_stopper.early_stop(epoch_loss):
                print(f"!!! 触发早停 (Loss 未下降持续 {early_stopper.patience} 轮) !!!")
                break

    except KeyboardInterrupt:
        print("\n\n>>> 训练被手动中断 (Ctrl+C) <<<")
        print("正在保存当前模型状态...")
        torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "interrupted.pth"))
        print("保存成功！文件位于: " + os.path.join(Config.SAVE_DIR, "interrupted.pth"))
        writer.close()
        sys.exit(0)

    print("训练正常结束。")
    writer.close()


if __name__ == "__main__":
    train()