import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils import *
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


# class MyDataset(Dataset):
#     def __init__(self, path):
#         self.path = path
#         self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
#
#     def __len__(self):
#         return len(self.name)
#
#     def __getitem__(self, index):
#         segment_name = self.name[index]  # xx.png
#         segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
#         image_path = os.path.join(self.path, 'JPEGImages', segment_name)
#         segment_image = keep_image_size_open(segment_path)
#         image = keep_image_size_open_rgb(image_path)
#         return transform(image), torch.Tensor(np.array(segment_image))


class MyDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.name = os.listdir(os.path.join(path, 'SegmentationClass'))
        # 预定义转换逻辑
        self.img_trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.name)

    def __getitem__(self, index):
        segment_name = self.name[index]
        name_without_ext = os.path.splitext(segment_name)[0]

        segment_path = os.path.join(self.path, 'SegmentationClass', segment_name)
        image_path = os.path.join(self.path, 'JPEGImages', name_without_ext + '.jpg')

        if not os.path.exists(image_path):
            image_path = os.path.join(self.path, 'JPEGImages', name_without_ext + '.png')

        # 1. 调用 utils 里的函数，得到固定尺寸 (512, 384) 的 PIL 图片
        segment_image = keep_image_size_open(segment_path, size=(512, 384))
        image = keep_image_size_open_rgb(image_path, size=(512, 384))

        # 2. 转换为 Tensor
        # image 变成 [C, H, W] 浮点型
        # label 变成 [H, W] 整型 (CrossEntropyLoss 需要 Long 类型)
        # 在 data.py 的 __getitem__ 结尾处修改
        segment_image = keep_image_size_open(segment_path, size=(512, 384))
        image = keep_image_size_open_rgb(image_path, size=(512, 384))

        # 将 PIL 转为 numpy 数组进行处理
        mask_np = np.array(segment_image)

        # 【核心修复代码】
        # 强制将所有大于等于 num_classes 的值（比如255）归为背景（0）
        # 或者你可以将其改为 0，确保索引在 [0, 2] 之间
        mask_np[mask_np >= 3] = 0

        return self.img_trans(image), torch.from_numpy(mask_np).long()
        return self.img_trans(image), torch.from_numpy(np.array(segment_image)).long()


#验证
# if __name__ == '__main__':
#     data = MyDataset('data')
#     img, mask = data[0]
#     print(f"图像形状: {img.shape}")
#     print(f"标签形状: {mask.shape}")
#     # 关键：看看标签里到底有哪些数字
#     print(f"标签中的所有类别数值: {torch.unique(mask)}")


if __name__ == '__main__':
    from torch.nn.functional import one_hot
    data = MyDataset('data')
    print(data[0][0].shape)
    print(data[0][1].shape)
    out=one_hot(data[0][1].long())
    print(out.shape)
