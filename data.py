import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from utils import keep_image_size_open, keep_image_size_open_rgb

# 定义原图的预处理：转为 Tensor 并归一化到 [0, 1]
transform_img = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        # --- 修改点：过滤 readme 和非图片文件 ---
        raw_names = os.listdir(os.path.join(path, 'SegmentationClass'))
        self.name = [
            f for f in raw_names
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))  # 只保留图片后缀
               and not f.lower().startswith('readme')  # 排除以 readme 开头的文件
        ]
        # 排序确保训练/验证集划分的稳定性
        self.name.sort()

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # 例如: "2010_005723.png"
        name_without_ext = os.path.splitext(segment_name)[0]

        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)

        # First search for .jpg, then search for .png
        image_path = os.path.join(self.path, 'JPEGImages', name_without_ext + '.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(self.path, 'JPEGImages', name_without_ext + '.png')

            # --- Robust：If can't find the file, try to skip ---
        if not os.path.exists(image_path):
            print(f"Warning: Can't find file {name_without_ext}，tried to skip this.")
            return self.__getitem__((index + 1) % len(self.name))

        # 1. 读取图片 (尺寸已经在 utils.py 中默认设为 512x384，这里显式传参更安全)
        img_pil = keep_image_size_open_rgb(image_path, size=(512, 384))
        mask_pil = keep_image_size_open(segment_path, size=(512, 384))

        # 2. 处理标签数据
        mask_np = np.array(mask_pil)

        # 【核心修复】防止类别越界，将大于等于 3 的边缘/杂色归为背景 0
        mask_np[mask_np >= 3] = 0

        # 3. 返回 Tensor
        return transform_img(img_pil), torch.from_numpy(mask_np).long()


if __name__ == '__main__':
    # 自测代码
    data = MyDataset('data')
    if len(data) > 0:
        img, mask = data[0]
        print(f"Effective data volume: {len(data)}")
        print(f"Image Shape: {img.shape}")
        print(f"Mask Shape: {mask.shape}")
        print(f"Unique classes in mask: {torch.unique(mask)}")
    else:
        print("No valid data found. Please check the path.")
