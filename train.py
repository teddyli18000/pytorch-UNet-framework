import os
import tqdm
from torch import nn, optim
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

# --- 导入你的工具包 ---
from tools import LossLogger, get_split_loaders

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'

if __name__ == '__main__':
    # --- 1. 数据与模型准备 ---
    num_classes = 2 + 1

    # 初始化原始数据集
    full_dataset = MyDataset(data_path)

    # 使用 tools.py 中的逻辑划分：训练集 90%，验证集 10%
    # RTX 5060 建议 batch_size 设为 8，如果显存报错请调至 4
    train_loader, val_loader = get_split_loaders(full_dataset, batch_size=8, train_ratio=0.9)

    net = UNet(num_classes).to(device)

    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print('成功加载权重文件！')
    else:
        print('未找到权重文件，开始从头训练')

    # --- 2. 优化器、损失函数与调度器 ---
    initial_lr = 1e-4  # 针对 1.8w 张图片，1e-4 是稳健的选择
    opt = optim.Adam(net.parameters(), lr=initial_lr)
    loss_fun = nn.CrossEntropyLoss()

    total_epochs = 50  # 建议改为 50 轮，1.8w 张图跑 100 轮可能过拟合
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=1e-7)

    scaler = torch.amp.GradScaler('cuda')

    # 初始化 Loss 记录器
    logger = LossLogger(save_path='unet_loss_curve.png')

    # --- 3. 训练主循环 ---
    epoch = 1
    while epoch <= total_epochs:
        net.train()
        total_train_loss = 0

        # 使用 train_loader 进行训练
        pbar = tqdm.tqdm(train_loader)
        for i, (image, segment_image) in enumerate(pbar):
            image, segment_image = image.to(device), segment_image.to(device)

            with torch.amp.autocast('cuda'):
                out_image = net(image)
                train_loss = loss_fun(out_image, segment_image.long())

            opt.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(opt)
            scaler.update()

            total_train_loss += train_loss.item()

            pbar.set_description(f"Epoch {epoch}/{total_epochs}")
            pbar.set_postfix(loss=f"{train_loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.7f}")

            # 预览图保存逻辑（每轮保存第一张）
            if i == 0:
                _segment_image = segment_image[0].detach().cpu().unsqueeze(0).float() / 2.0
                _out_image = torch.argmax(out_image[0], dim=0).detach().cpu().unsqueeze(0).float() / 2.0
                img = torch.stack([_segment_image, _out_image], dim=0)
                save_image(img, f'{save_path}/epoch_{epoch}_preview.png', normalize=False)

        # --- 每个 Epoch 结束的操作 ---
        avg_loss = total_train_loss / len(train_loader)

        # 记录 Loss 并更新图表
        logger.update(avg_loss)

        scheduler.step()

        # 针对 1.8w 图片，建议每 5 轮存一次，防止崩溃丢失进度
        if epoch % 5 == 0:
            if not os.path.exists('params'): os.makedirs('params')
            current_weight_path = f'params/unet_epoch_{epoch}.pth'
            torch.save(net.state_dict(), current_weight_path)

            # # 同时保留一个最新的权重文件，方便断点续传（可选）
            # torch.save(net.state_dict(), weight_path)

            print(f'\n[已保存] 第 {epoch} 轮模型: {current_weight_path}')
        epoch += 1