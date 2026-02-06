import os
import tqdm
from torch import nn, optim
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

# --- 导入工具包 ---
from tools import LossLogger, get_split_loaders

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet_resume.pth'  # 用于恢复训练的最新断点文件
data_path = r'data'
save_path = 'train_image'

if __name__ == '__main__':
    # --- 1. 参数设置 ---
    num_classes = 2 + 1
    total_epochs = 50
    initial_lr = 1e-4

    # 梯度累加设置：真实 BatchSize = batch_size * accumulation_steps
    # 显存占用看 batch_size (设为 2 非常安全)，训练效果看等效 BatchSize (2 * 4 = 8)
    batch_size = 4
    accumulation_steps = 4

    # --- 2. 数据与模型准备 ---
    full_dataset = MyDataset(data_path)
    train_loader, val_loader = get_split_loaders(full_dataset, batch_size=batch_size, train_ratio=0.9)

    net = UNet(num_classes).to(device)
    opt = optim.Adam(net.parameters(), lr=initial_lr)
    loss_fun = nn.CrossEntropyLoss()
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=1e-7)
    scaler = torch.amp.GradScaler('cuda')
    logger = LossLogger(save_path='unet_loss_curve.png')

    # --- 3. 断点续传加载逻辑 ---
    start_epoch = 1
    if os.path.exists(weight_path):
        # 注意：加载完整断点包需设置 weights_only=False
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

        net.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1

        if 'logger_train_losses' in checkpoint:
            logger.train_losses = checkpoint['logger_train_losses']

        print(f'成功恢复训练！接上次进度，从第 {start_epoch} 轮开始...')
    else:
        print('未找到断点文件，开始全新训练')

    # --- 4. 训练主循环 ---
    epoch = start_epoch
    while epoch <= total_epochs:
        net.train()
        total_train_loss = 0
        opt.zero_grad()  # 梯度清零放在循环外（配合累加）

        pbar = tqdm.tqdm(train_loader)
        for i, (image, segment_image) in enumerate(pbar):
            image, segment_image = image.to(device), segment_image.to(device)

            with torch.amp.autocast('cuda'):
                out_image = net(image)
                # 损失值除以累加步数进行归一化
                train_loss = loss_fun(out_image, segment_image.long()) / accumulation_steps

            # 反向传播，累积梯度
            scaler.scale(train_loss).backward()

            # 当达到累加步数，或者到了数据集最后一份，执行优化
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

            total_train_loss += train_loss.item() * accumulation_steps  # 还原真实Loss显示

            pbar.set_description(f"Epoch {epoch}/{total_epochs}")
            pbar.set_postfix(loss=f"{train_loss.item() * accumulation_steps:.4f}",
                             lr=f"{opt.param_groups[0]['lr']:.7f}")

            # 预览图保存
            if i == 0:
                _segment_image = segment_image[0].detach().cpu().unsqueeze(0).float() / 2.0
                _out_image = torch.argmax(out_image[0], dim=0).detach().cpu().unsqueeze(0).float() / 2.0
                img = torch.stack([_segment_image, _out_image], dim=0)
                save_image(img, f'{save_path}/epoch_{epoch}_preview.png', normalize=False)

        # --- 每个 Epoch 结束的操作 ---
        avg_loss = total_train_loss / len(train_loader)
        logger.update(avg_loss)
        scheduler.step()

        # --- 每一轮保存完整断点 (断点续传核心) ---
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'logger_train_losses': logger.train_losses
        }

        if epoch % 5 == 0:
            if not os.path.exists('params'): os.makedirs('params')
            current_weight_path = f'params/unet_epoch_{epoch}.pth'
            torch.save(checkpoint_data, current_weight_path)
            print(f'\n[已存档] 第 {epoch} 轮模型已保存至 {current_weight_path}')

        if not os.path.exists('params'): os.makedirs('params')
        torch.save(checkpoint_data, weight_path)  # master断点始终更新

        epoch += 1