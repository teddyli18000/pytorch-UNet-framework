# Pytorch-UNet-framework (Archived)

### Introduction

    Build your own Unet network with PyTorch and train it on your own dataset.

### Current version

* **v3.0 - feat:** Optimize the selection when saving result

### Underlying framework

    Python 3.13

    torch 2.9.1+cu128 ->2.10.0

    torchvision 0.24.1+cu128 ->0.25.0

### Data(In my Baidu NetDisk)

pytorch-UNet-framework

### Instruction Manual

1.  #### Prepare dataset:
    #### Storage address of the original dataset images：
        data/JPEGImages
    #### Storage address of the masks：
        data/SegmentationClass
2. #### Train:
       train.py
3. #### Save path:

   ##### (1).The "train_image" folder stores
       Effect images generated during the training process
   ##### (2).The "params" folder stores
       Weights
4. #### Test
       Run test_single.py for one images (Supports custom/source save paths), 
       Run test_multiple.py for multiple images in one folder.

       Program store the results with the suffix "_predict".

### Historical version

#### For major version update history, see ["version-log.md"](./version-log.md)

- **v3.0 - feat:** Optimize the selection when saving result

- **v2.3 - fix:** Weight matching error when loading .pth
- **v2.2 - refactor:** Loss-curve for single training round & gitignore

- **v2.1 - fix:** "readme.md" mixed in during data processing

* **v2.0** - Add resume train function

* **v1.0** - Main structure

> [Back to Top](#Pytorch-UNet-framework)