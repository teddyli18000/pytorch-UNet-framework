import os
import cv2
import numpy as np
import torch
from net import UNet
from utils import keep_image_size_open_rgb
from torchvision import transforms

# 1. 确保和训练时设置一致
transform = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(3).to(device)  # 3分类

weights = 'params/unet.pth'
if os.path.exists(weights):
    # 加上 map_location 确保在 CPU/GPU 都能跑
    net.load_state_dict(torch.load(weights, map_location=device))
    print('Successfully loaded weights.')
else:
    print('No weights found!!!')

# 2. 输入图片
_input = input('Please input JPEGImages path (e.g., data/JPEGImages/xxx.jpg): ')

# 3. 预处理 (必须指定和训练时一样的尺寸 512x384)
img = keep_image_size_open_rgb(_input, size=(512, 384))
img_data = transform(img).to(device)
img_data = torch.unsqueeze(img_data, dim=0)  # [1, 3, 384, 512]

# 4. 推理
net.eval()
with torch.no_grad():
    out = net(img_data)  # [1, 3, 384, 512]
    out = torch.argmax(out, dim=1)  # [1, 384, 512]
    out = torch.squeeze(out, dim=0)  # [384, 512]

print(f"Predicted classes: {set(out.reshape(-1).tolist())}")

# 5. 可视化保存
# 因为类别是 0, 1, 2，为了人眼能看清，我们要拉伸像素值
# 0->0(黑), 1->127(灰), 2->255(白)
out_np = out.cpu().numpy().astype(np.uint8)
out_np = out_np * 127  # 简单可视化映射

# 保存到 result 文件夹
result_dir = 'result'
os.makedirs(result_dir, exist_ok=True)

base_name = os.path.basename(_input)
name_without_ext = os.path.splitext(base_name)[0]
save_path = os.path.join(result_dir, f"{name_without_ext}_predict.png")

cv2.imwrite(save_path, out_np)
print(f"Result saved to {save_path}")

#
# #save to source folder
# # 处理文件名
# dir_name = os.path.dirname(_input)                  # 输入图片所在文件夹
# base_name = os.path.basename(_input)               # 原文件名，例如 "image1.jpg"
# name_without_ext = os.path.splitext(base_name)[0]  # 去掉扩展名，例如 "image1"
# save_name = f"{name_without_ext}_output.png"       # 拼接成 "image1_output.png"
#
# # 保存路径
# save_path = os.path.join(dir_name, save_name)
#
# # 保存结果
# cv2.imwrite(save_path, out_np)
# print(f"Result saved to {save_path}")
