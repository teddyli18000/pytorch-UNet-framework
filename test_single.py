import os
import cv2
import numpy as np
import torch
from net import UNet
from utils import keep_image_size_open_rgb
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = UNet(3).to(device)  # 3分类

weights = 'params/unet.pth'
if os.path.exists(weights):
    # 加上 map_location 确保在 CPU/GPU 都能跑
    # 1. 先加载完整的断点字典
    checkpoint = torch.load(weights, map_location=device, weights_only=False)

    # 2. 从字典中提取出模型权重部分，再加载进网络
    net.load_state_dict(checkpoint['model_state_dict'])
    print('Successfully loaded weights.')
else:
    print('No weights found!!!')
    exit()

transform = transforms.Compose([
    transforms.ToTensor()
])

print("(Default)0: Output to default directory (result), (You can press Enter directly)")
print("1: Output to specified directory (you will enter in the next step)")
print("2: Output to source directory")

status_output = input("""Please enter the "output status"[0/1/2]:""").strip() or "0"

target_directory = ""
if status_output == "1":
    target_directory = input("""Please enter the "output directory":""")

# 2. 输入图片
_input_origin = input('Please input JPEGImages path (e.g., data/JPEGImages/xxx.jpg): ')

# 3. 预处理 (必须指定和训练时一样的尺寸 512x384)
img = keep_image_size_open_rgb(_input_origin, size=(512, 384))
img_data = transform(img).to(device)
img_data = torch.unsqueeze(img_data, dim=0)  # [1, 3, 384, 512]

# 4. 推理
net.eval()
with torch.no_grad():
    out = net(img_data)  # [1, 3, 384, 512]
    out = torch.argmax(out, dim=1)  # [1, 384, 512]
    out = torch.squeeze(out, dim=0)  # [384, 512]

print(f"Verify: Predicted classes: {set(out.reshape(-1).tolist())}")

# 5. 可视化保存
# 因为类别是 0, 1, 2，为了人眼能看清，我们要拉伸像素值
# 0->0(黑), 1->127(灰), 2->255(白)
out_np = out.cpu().numpy().astype(np.uint8)
out_np = out_np * 127  # 简单可视化映射

if status_output == "0":  # save to result directory (Default)
    result_dir = 'result'
    os.makedirs(result_dir, exist_ok=True)

    base_name = os.path.basename(_input_origin)
    name_without_ext = os.path.splitext(base_name)[0]
    save_path = os.path.join(result_dir, f"{name_without_ext}_predict.png")

    cv2.imwrite(save_path, out_np)
    print(f"Result saved to {save_path}")

elif status_output == "1":  # save to specified directory
    result_dir = target_directory

    base_name = os.path.basename(_input_origin)
    name_without_ext = os.path.splitext(base_name)[0]
    save_path = os.path.join(result_dir, f"{name_without_ext}_predict.png")

    cv2.imwrite(save_path, out_np)
    print(f"Result saved to {save_path}")

elif status_output == "2":  # save to source folder
    result_dir = os.path.dirname(_input_origin)

    base_name = os.path.basename(_input_origin)
    name_without_ext = os.path.splitext(base_name)[0]
    save_path = os.path.join(result_dir, f"{name_without_ext}_predict.png")

    cv2.imwrite(save_path, out_np)
    print(f"Result saved to {save_path}")
