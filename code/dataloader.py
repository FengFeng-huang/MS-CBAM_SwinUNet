import os
import random
import numpy as np
from PIL import Image
from PIL.Image import Resampling
import tifffile
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter, map_coordinates


class ImageMaskDataset(Dataset):
    def __init__(self,image_dir,mask_dir,dataset_name,
                 augment=False,transform_image=None,transform_mask=None,size=(224, 224)
                 ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.dataset_name = dataset_name
        self.transform_image = transform_image
        self.transform_mask = transform_mask
        self.size = size
        self.augment = augment
        print(f"The augmentation option is {self.augment}")
        # save suffix
        exts = ('.jpg', '.png', '.tif', '.tiff')
        self.images = sorted([
            fname for fname in os.listdir(image_dir)
            if fname.lower().endswith(exts)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img_path = os.path.join(self.image_dir, fname)
        mask_path = os.path.join(self.mask_dir, fname)

        # 1) 读图、读 mask
        if fname.lower().endswith(('.tif', '.tiff')):
            image_np = tifffile.imread(img_path)
            img = Image.fromarray(image_np).convert("RGB")
            mask_np = tifffile.imread(mask_path)
            msk = Image.fromarray(mask_np).convert("L")
        else:
            img = Image.open(img_path).convert("RGB")
            msk = Image.open(mask_path).convert("L")

        # 2) Resize：图用双线性，mask 用最近邻
        img = img.resize(self.size, resample=Resampling.BILINEAR)
        msk = msk.resize(self.size, resample=Resampling.NEAREST)

        # 3) 数据增强
        if self.augment:
            img, msk = self.apply_augment(img, msk)

        # 4) 如果有自定义的 PIL-level augment
        if self.transform_image:
            img = self.transform_image(img)
        if self.transform_mask:
            msk = self.transform_mask(msk)

        # 5) 转成 numpy → Tensor
        img = np.array(img, dtype=np.float32) / 255.0     # [H,W,3], 0~1
        img = np.transpose(img, (2, 0, 1))               # [3,H,W]
        img_t = torch.from_numpy(img)

        msk = np.array(msk, dtype=np.int64)              # [H,W], 整数
        if self.dataset_name != 'Synapse':
            msk = np.where(msk <= 160, 0, 1).astype(np.int64)
        msk_t = torch.from_numpy(msk)                    # LongTensor

        return img_t, msk_t

    def apply_augment(self, img, msk):

        '''
        # 随机水平翻转
        if random.random() < 0.2:
            img = TF.hflip(img)
            msk = TF.hflip(msk)
        '''

        # 随机垂直翻转
        if random.random() < 0.2:
            img = TF.vflip(img)
            msk = TF.vflip(msk)
        # 随机旋转和放大（仿射变换）
        if random.random() < 0.2:
            # 只旋转，不缩放
            angle = random.uniform(-20, 20)
            img = TF.rotate(img, angle, interpolation=Resampling.BILINEAR)
            msk = TF.rotate(msk, angle, interpolation=Resampling.NEAREST)

        # —— 3) 随机放大
        if random.random() < 0.4:
            scale = random.uniform(1.0, 1.1)
            # 这里用 affine 做纯放大（angle=0, shear=0, translate=(0,0)）
            img = TF.affine(
                img,
                angle=0, translate=(0, 0), scale=scale, shear=0,
                interpolation=Resampling.BILINEAR
            )
            msk = TF.affine(
                msk,
                angle=0, translate=(0, 0), scale=scale, shear=0,
                interpolation=Resampling.NEAREST
            )

        # 弹性形变
        #if random.random() < 0.5:
            #img, msk = self.elastic_transform(img, msk)

        return img, msk

    def elastic_transform(self, img, msk, alpha=100, sigma=8, mode='reflect'):
        # PIL → numpy
        img_np = np.array(img)
        msk_np = np.array(msk)
        H, W = img_np.shape[:2]

        # 1) 随机位移场
        dx = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma, mode=mode) * alpha
        dy = gaussian_filter((np.random.rand(H, W) * 2 - 1), sigma, mode=mode) * alpha
        x, y = np.meshgrid(np.arange(W), np.arange(H))

        # 2) 构造索引，注意 ravel 后是一维长度 H*W
        indices = ( (y + dy).ravel(), (x + dx).ravel() )

        # 3) 对每个通道分别做 map_coordinates，再 reshape 回 (H, W)
        img_deformed = np.zeros_like(img_np)
        for c in range(img_np.shape[2]):
            channel = img_np[..., c]
            warped = map_coordinates(channel, indices, order=1, mode=mode)
            img_deformed[..., c] = warped.reshape(H, W)

        # 4) mask 用最近邻插值
        msk_deformed = map_coordinates(msk_np, indices, order=0, mode=mode).reshape(H, W)

        # 5) numpy → PIL
        img_out = Image.fromarray(img_deformed.astype(np.uint8))
        msk_out = Image.fromarray(msk_deformed.astype(np.uint8))

        return img_out, msk_out

def check_dataloader(dataloader):
    num_batches = len(dataloader)
    print(f'Number of batches: {num_batches}')

    for data, target in dataloader:
        print(f'Data shape: {data.shape}')
        print(f'Target shape: {target.shape}')
        break