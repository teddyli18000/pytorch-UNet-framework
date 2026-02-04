import os
import cv2
import numpy as np
import torch

from net import *
from utils import *
from data import *
from torchvision.utils import save_image
from PIL import Image

# 初始化网络
net = UNet(3).cuda()

weights = 'params/unet.pth'
if os.path.exists(weights):
    net.load_state_dict(torch.load(weights))
    print('Successfully loaded weights.')
else:
    print('No weights loaded.')

# 输入图片路径
_input = input('Please input JPEGImages path: ')

# 处理图片
img = keep_image_size_open_rgb(_input)
img_data = transform(img).cuda()
img_data = torch.unsqueeze(img_data, dim=0)

# 推理
net.eval()
with torch.no_grad():
    out = net(img_data)
    out = torch.argmax(out, dim=1)
    out = torch.squeeze(out, dim=0)
    out = out.unsqueeze(dim=0)

# 打印类别集合
print(set(out.reshape(-1).tolist()))

# 转成 numpy 并归一化
out_np = out.permute(1, 2, 0).cpu().numpy() * 255.0
out_np = out_np.astype(np.uint8)

# 创建 result 文件夹（如果不存在）
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)
# save to result fold
# # 处理文件名
# base_name = os.path.basename(_input)                  # 获取文件名，例如 "image1.jpg"
# name_without_ext = os.path.splitext(base_name)[0]    # 去掉扩展名，例如 "image1"
# save_name = f"{name_without_ext}_output.png"         # 拼接成 "image1_output.png"
#
# # 创建 result 文件夹（如果不存在）
# result_dir = 'result'
# os.makedirs(result_dir, exist_ok=True)
#
# # 保存结果
# save_path = os.path.join(result_dir, save_name)
# cv2.imwrite(save_path, out_np)
# print(f"Result saved to {save_path}")


#save to source folder
# 处理文件名
dir_name = os.path.dirname(_input)                  # 输入图片所在文件夹
base_name = os.path.basename(_input)               # 原文件名，例如 "image1.jpg"
name_without_ext = os.path.splitext(base_name)[0]  # 去掉扩展名，例如 "image1"
save_name = f"{name_without_ext}_output.png"       # 拼接成 "image1_output.png"

# 保存路径
save_path = os.path.join(dir_name, save_name)

# 保存结果
cv2.imwrite(save_path, out_np)
print(f"Result saved to {save_path}")