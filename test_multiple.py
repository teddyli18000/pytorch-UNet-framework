import os
import cv2
import numpy as np
import torch
from net import UNet
from utils import keep_image_size_open_rgb
from torchvision import transforms
from tqdm import tqdm  # pip install tqdm

# --- 1. 环境与模型准备 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(3).to(device)  # 3分类

weights = 'params/unet.pth'
if os.path.exists(weights):
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    net.load_state_dict(checkpoint['model_state_dict'])
    print('Successfully loaded weights.')
else:
    print('No weights found!!!')
    exit()

transform = transforms.Compose([transforms.ToTensor()])
net.eval()

# --- 2. 输入配置 ---
input_dir_path = input('Please input JPEGImages FOLDER path (e.g., data/JPEGImages/): ')
if not os.path.isdir(input_dir_path):
    print("Error: The input path is not a directory!")
    exit()

print("\n(Default)0: Output to 'result' folder")
print("1: Output to specified directory")
print("2: Output to source directory")
status_output = input("""Please enter output status [0/1/2]: """).strip() or "0"

target_dir = ""
if status_output == "1":
    target_dir = input('Please enter the "output directory": ')
    os.makedirs(target_dir, exist_ok=True)
elif status_output == "0":
    target_dir = 'result'
    os.makedirs(target_dir, exist_ok=True)

# 获取文件夹内所有图片（支持 jpg, png, jpeg）
img_extensions = ('.jpg', '.png', '.jpeg', '.JPG', '.PNG')
all_files = [f for f in os.listdir(input_dir_path) if f.endswith(img_extensions)]

print(f"Found {len(all_files)} images. Starting inference...")

# --- 3. 循环处理 ---
for filename in tqdm(all_files):
    # 拼接完整输入路径
    img_path = os.path.join(input_dir_path, filename)

    # 预处理
    img = keep_image_size_open_rgb(img_path, size=(512, 384))
    img_data = transform(img).to(device)
    img_data = torch.unsqueeze(img_data, dim=0)

    # 推理
    with torch.no_grad():
        out = net(img_data)
        out = torch.argmax(out, dim=1)
        out = torch.squeeze(out, dim=0)

    # 转换与可视化 (0->0, 1->127, 2->254)
    out_np = out.cpu().numpy().astype(np.uint8)
    out_np = out_np * 127

    # 确定保存路径
    if status_output == "2":
        save_dir = input_dir_path
    else:
        save_dir = target_dir

    name_without_ext = os.path.splitext(filename)[0]
    save_path = os.path.join(save_dir, f"{name_without_ext}_predict.png")

    # 保存
    cv2.imwrite(save_path, out_np)

print(f"\nDone! All results save to: {target_dir}.")
