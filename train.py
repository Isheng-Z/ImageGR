# train.py
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm

from config import Config
from dataset import get_dataloader
from full_model import RSS_Flow_Net


def train():
    # 1. 初始化
    print(f"正在初始化 RSS-Flow (Step 2)... Device: {Config.DEVICE}")
    os.makedirs(Config.SAVE_DIR, exist_ok=True)
    os.makedirs(Config.LOG_DIR, exist_ok=True)

    writer = SummaryWriter(Config.LOG_DIR)
    train_loader = get_dataloader(train=True)

    # 2. 构建模型
    model = RSS_Flow_Net().to(Config.DEVICE)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # 3. 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)

    # 4. 训练循环
    global_step = 0
    for epoch in range(Config.EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{Config.EPOCHS}")

        for x1, labels in progress_bar:
            x1 = x1.to(Config.DEVICE)  # 真实图像 (Target)
            labels = labels.to(Config.DEVICE)
            B = x1.shape[0]

            # === Flow Matching 核心逻辑 ===
            # x0: 高斯噪声 (Source)
            x0 = torch.randn_like(x1).to(Config.DEVICE)

            # t: 随机时间步 [0, 1]
            t = torch.rand(B, device=Config.DEVICE)

            # 插值路径: x_t = t*x1 + (1-t)*x0
            # 广播 t 以匹配图像维度
            t_view = t.view(B, 1, 1, 1)
            x_t = t_view * x1 + (1 - t_view) * x0

            # 目标速度: v = x1 - x0 (指向真实图像的方向)
            target_v = x1 - x0

            # === 模型前向传播 ===
            # 预测速度 v_pred
            pred_v = model(x_t, t, labels)

            # === 损失计算 ===
            loss = F.mse_loss(pred_v, target_v)

            # === 反向传播 ===
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # === 记录 ===
            running_loss = loss.item()
            progress_bar.set_postfix(loss=running_loss)

            if global_step % Config.LOG_INTERVAL == 0:
                writer.add_scalar('Loss/train', running_loss, global_step)

                # 监控 Router 的行为 (可选，打印第一个样本的Gate)
                # with torch.no_grad():
                #     gates = model.rss_module.router(labels[:1])
                #     writer.add_histogram('Router/Gates', gates, global_step)

            global_step += 1

        # === 每个 Epoch 保存一次模型 ===
        save_path = os.path.join(Config.SAVE_DIR, f"rss_flow_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), save_path)
        print(f"模型已保存: {save_path}")

    writer.close()
    print("训练完成！")


if __name__ == "__main__":
    train()