# Pytorch-UNet-framework

### Introduction

    Build your own Unet network with PyTorch and train it on your own dataset.

### Current version

    v2.3 fix bugs: Weight matching error when loading .pth

### Underlying framework

    Python 3.13

    torch 2.9.1+cu128 ->2.10.0

    torchvision 0.24.1+cu128 ->0.25.0

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
       Run test.py file to test images.
       Program store the test results in the source folder with the suffix "_predict".

### Historical version

    v2.3 fix bugs: Weight matching error when loading .pth
    v2.2 optimize: loss-curve for single training round & gitignore
    v2.1 fix bugs: readme.md mixed in during data processing
    v2.0 Add "resume from breakpoint" function

    v1.0 Basic framework