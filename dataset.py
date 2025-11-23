# dataset.py
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import Config


def get_dataloader(train=True):
    """
    获取 CIFAR-10 数据加载器
    """
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # 将图像归一化到 [-1, 1]，符合 Flow Matching 的习惯
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=train,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True
    )

    return dataloader