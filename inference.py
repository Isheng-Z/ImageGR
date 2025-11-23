# inference.py
import torch
import torchvision
import os
from config import Config
from full_model import RSS_Flow_Net


def generate_images(checkpoint_path, num_steps=50):
    device = Config.DEVICE
    print(f"加载模型: {checkpoint_path}")

    model = RSS_Flow_Net().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 准备采样条件
    n_samples = 16
    # 生成 0-9 类的样本
    y = torch.tensor([i % 10 for i in range(n_samples)]).to(device)

    # 起点: 纯噪声 (x0)
    x = torch.randn(n_samples, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(device)

    # Euler ODE Solver (最简单的流匹配采样)
    dt = 1.0 / num_steps

    print("开始采样 (Neural ODE)...")
    with torch.no_grad():
        for i in range(num_steps):
            # 当前时间 t
            t = torch.ones(n_samples, device=device) * (i * dt)

            # 预测速度 v
            v = model(x, t, y)

            # 更新位置 x_new = x_old + v * dt
            x = x + v * dt

    # 反归一化: [-1, 1] -> [0, 1]
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)

    # 保存结果
    os.makedirs("./results", exist_ok=True)
    save_path = "./results/generated_sample.png"
    torchvision.utils.save_image(x, save_path, nrow=4)
    print(f"生成图像已保存至: {save_path}")


if __name__ == "__main__":
    # 自动寻找最新的权重
    if os.path.exists(Config.SAVE_DIR):
        checkpoints = sorted(os.listdir(Config.SAVE_DIR))
        if checkpoints:
            latest_ckpt = os.path.join(Config.SAVE_DIR, checkpoints[-1])
            generate_images(latest_ckpt)
        else:
            print("未找到权重文件，请先运行 train.py")
    else:
        print("未找到权重目录")