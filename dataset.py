import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from config import Config


def get_dataloader(train=True):
    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
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
        pin_memory=True,
        # [优化] 保持 Worker 进程存活，避免每个 Epoch 重新创建的开销
        persistent_workers=(Config.NUM_WORKERS > 0),
        drop_last=True
    )

    return dataloader