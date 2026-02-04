from PIL import Image

def keep_image_size_open(path, size=(512, 384)):
    """
    处理标签 (Mask)：必须使用最近邻插值 (Image.NEAREST)，
    防止改变像素值类别（例如把 1 和 2 插值成 1.5）。
    """
    img = Image.open(path)
    temp = max(img.size)
    # 创建正方形背景，标签通常用 P 模式，背景给 0
    mask = Image.new('P', (temp, temp), 0)
    mask.paste(img, (0, 0))
    # 【关键修改】必须加 resample=Image.NEAREST
    mask = mask.resize(size, resample=Image.NEAREST)
    return mask

def keep_image_size_open_rgb(path, size=(512, 384)):
    """
    处理原图 (RGB)：可以使用双三次或双线性插值，让图片更平滑。
    """
    img = Image.open(path)
    temp = max(img.size)
    # 创建正方形背景，RGB 模式，背景给黑色 (0,0,0)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    mask.paste(img, (0, 0))
    # 原图可以用平滑插值
    mask = mask.resize(size, resample=Image.BICUBIC)
    return mask