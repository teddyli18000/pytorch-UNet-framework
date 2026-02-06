import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split


class LossLogger:
    """训练监控工具：记录 Loss 并自动绘图"""

    def __init__(self, save_path='loss_curve.png'):
        self.train_losses = []
        self.val_losses = []
        self.save_path = save_path

    def update(self, train_loss, val_loss=None):
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        self._plot()

    def _plot(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss', color='blue', lw=2)
        if self.val_losses:
            plt.plot(self.val_losses, label='Val Loss', color='red', linestyle='--', lw=2)
        plt.title('UNet Training Process')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(self.save_path)
        plt.close()


def get_split_loaders(dataset, batch_size=8, train_ratio=0.9):
    """
    逻辑划分数据集：
    1. 不移动物理文件
    2. 随机划分 18000 张图片
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    # 随机打乱索引并按比例划分
    train_idx, val_idx = train_test_split(
        indices,
        train_size=train_ratio,
        shuffle=True,
        random_state=42
    )

    # 创建子集视图
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    # 创建对应的 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"数据划分完成！总计: {dataset_size} | 训练集: {len(train_idx)} | 验证集: {len(val_idx)}")
    return train_loader, val_loader
