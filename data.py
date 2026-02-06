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
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]  # 例如: "2010_005723.png"
        name_without_ext = os.path.splitext(segment_name)[0]

        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)

        # 优先找 .jpg，找不到找 .png
        image_path = os.path.join(self.path, 'JPEGImages', name_without_ext + '.jpg')
        if not os.path.exists(image_path):
            image_path = os.path.join(self.path, 'JPEGImages', name_without_ext + '.png')

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
    img, mask = data[0]
    print(f"Image Shape: {img.shape}")
    print(f"Mask Shape: {mask.shape}")
    print(f"Unique classes in mask: {torch.unique(mask)}")
