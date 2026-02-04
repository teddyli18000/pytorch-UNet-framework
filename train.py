import os
import tqdm
from torch import nn, optim
import torch
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path = 'params/unet.pth'
data_path = r'data'
save_path = 'train_image'

if __name__ == '__main__':
    # --- 1. 数据与模型准备 ---
    num_classes = 2 + 1
    data_loader = DataLoader(MyDataset(data_path), batch_size=8, shuffle=True)
    net = UNet(num_classes).to(device)

    if os.path.exists(weight_path):
        # 建议加上 weights_only=True 以符合新版本安全要求
        net.load_state_dict(torch.load(weight_path, map_location=device, weights_only=True))
        print('successful load weight！')
    else:
        print('not successful load weight')

    # --- 2. 优化器、损失函数与调度器 ---
    initial_lr = 1e-4
    opt = optim.Adam(net.parameters(), lr=initial_lr)
    loss_fun = nn.CrossEntropyLoss()

    total_epochs = 100
    scheduler = lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=1e-7)

    # 适配新版 PyTorch 的 Scaler (对应 Python 3.13)
    scaler = torch.amp.GradScaler('cuda')

    # --- 3. 训练主循环 ---
    epoch = 1
    while epoch <= total_epochs:
        net.train()
        # 使用 pbar 包裹可以在进度条右侧看到动态数据
        pbar = tqdm.tqdm(data_loader)
        for i, (image, segment_image) in enumerate(pbar):
            image, segment_image = image.to(device), segment_image.to(device)

            # --- 混合精度训练 (AMP) 修正为新版写法 ---
            with torch.amp.autocast('cuda'):
                out_image = net(image)
                train_loss = loss_fun(out_image, segment_image.long())

            opt.zero_grad()
            scaler.scale(train_loss).backward()
            scaler.step(opt)
            scaler.update()

            # 更新进度条显示的后缀
            pbar.set_description(f"Epoch {epoch}/{total_epochs}")
            pbar.set_postfix(loss=f"{train_loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.7f}")

            # 预览图保存逻辑
            # --- 修改后的保存逻辑 ---
            if i == 0:
                # 1. 提取第一张图并转为 Float (非常重要)
                _segment_image = segment_image[0].detach().cpu().unsqueeze(0).float()
                _out_image = torch.argmax(out_image[0], dim=0).detach().cpu().unsqueeze(0).float()

                # 2. 归一化到 [0, 1] 范围，这样 save_image 内部处理就不会报错
                # 因为你有 3 类 (0, 1, 2)，除以 2 就能把值变成 0, 0.5, 1.0
                _segment_image = _segment_image / 2.0
                _out_image = _out_image / 2.0

                # 3. 堆叠并保存
                img = torch.stack([_segment_image, _out_image], dim=0)
                # 加上 normalize=False 避免 torchvision 再次乱动你的像素值
                save_image(img, f'{save_path}/epoch_{epoch}_preview.png', normalize=False)

        # --- 每个 Epoch 结束的操作 ---
        scheduler.step()  # 必须在每个 epoch 结束调用

        if epoch % 20 == 0:
            if not os.path.exists('params'): os.makedirs('params')
            torch.save(net.state_dict(), weight_path)
            print(f'\nModel saved successfully at epoch {epoch}!')

        epoch += 1