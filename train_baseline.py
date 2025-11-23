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

# 导入配置和数据
from config import Config
from dataset import get_dataloader
# 导入刚才定义的纯净模型
from sit_baseline_model import Pure_SiT_Net
from torch.amp import GradScaler

warnings.filterwarnings("ignore")

# 临时覆盖 Config 中的参数，确保 Baseline 有足够的容量
# 你也可以去 config.py 里改，但在这里改更安全，不影响主实验
Config.SIT_HIDDEN_SIZE = 128
Config.SIT_DEPTH = 6
Config.SIT_HEADS = 4
Config.BATCH_SIZE = 256
Config.LR = 3e-4  # 标准 Transformer 学习率


@torch.no_grad()
def sample_images(model, device, epoch, writer, num_samples=8):
    model.eval()
    y = torch.randint(0, Config.NUM_CLASSES, (num_samples,), device=device)
    x = torch.randn(num_samples, 3, Config.IMG_SIZE, Config.IMG_SIZE, device=device)

    # 标准采样 50 步
    num_steps = 50
    dt = 1.0 / num_steps

    for i in range(num_steps):
        t = torch.ones(num_samples, device=device) * (i * dt)
        v = model(x, t, y)  # 这里不需要接收 gates
        x = x + v * dt

    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    grid = torchvision.utils.make_grid(x, nrow=4)
    writer.add_image('Baseline/Generated', grid, epoch)
    model.train()


def train():
    torch.set_float32_matmul_precision('high')
    if Config.CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True

    print(f"[{Config.DEVICE}] 启动纯 SiT 基线训练 (No RSS)")
    print(f"Batch: {Config.BATCH_SIZE} | Hidden: {Config.SIT_HIDDEN_SIZE} | Depth: {Config.SIT_DEPTH}")

    log_dir = "./logs/baseline_sit"
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    train_loader = get_dataloader(train=True)
    model = Pure_SiT_Net().to(Config.DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    # Cosine Scheduler 对于打破 0.2 Loss 瓶颈非常重要
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    scaler = GradScaler('cuda')

    global_step = 0
    start_time = time.time()

    for epoch in range(Config.EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", file=sys.stdout)
        running_loss = 0.0

        for x1, labels in progress_bar:
            x1 = x1.to(Config.DEVICE, non_blocking=True)
            labels = labels.to(Config.DEVICE, non_blocking=True)
            B = x1.shape[0]

            x0 = torch.randn_like(x1)
            t = torch.rand(B, device=Config.DEVICE)
            t_view = t.view(B, 1, 1, 1)
            x_t = t_view * x1 + (1 - t_view) * x0
            target_v = x1 - x0

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                pred_v = model(x_t, t, labels)
                loss = F.mse_loss(pred_v, target_v)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if global_step % 50 == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)
            global_step += 1

        scheduler.step()  # 更新学习率

        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1} | Loss: {epoch_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            sample_images(model, Config.DEVICE, epoch + 1, writer)
            torch.save(model.state_dict(), os.path.join(Config.SAVE_DIR, "baseline_latest.pth"))

    writer.close()


if __name__ == "__main__":
    train()