import numpy as np
from torch.utils.data import DataLoader
from dataloader import ImageMaskDataset, check_dataloader
import matplotlib.pyplot as plt

color_mapping = {
    0: (0,   0,   0  ),   # 背景（可选）
    1: (255, 255, 255),   # 白色
    2: (255, 182, 193),   # 浅粉红
    3: (255, 0,   0  ),   # 红色
    4: (0,   255, 0  ),   # 绿色
    5: (0,   0,   255),   # 蓝色
    6: (255, 255, 0  ),   # 黄色
    7: (255, 0,   255),   # 品红
    8: (0,   255, 255)    # 青色
}

def label2rgb(mask, mapping):
    """把 H×W 的标签图转成 H×W×3 的 RGB 图."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in mapping.items():
        rgb[mask == label] = color
    return rgb

if __name__ == '__main__':
    # Check images in the dataset
    synapse_dataset = ImageMaskDataset(image_dir="../datasets/Synapse/train/images", mask_dir="../datasets/Synapse/train/masks",
                                       dataset_name='Synapse', augment=False)
    synapse_train_loader = DataLoader(synapse_dataset, batch_size=1, shuffle=True)

    print("Train Loader:")
    check_dataloader(synapse_train_loader)

    # Show the image
    image, mask = next(iter(synapse_train_loader))
    image = image[0].permute(1, 2, 0).numpy()
    mask = mask[0].numpy()
    print(np.unique(mask))
    mask_rgb = label2rgb(mask, color_mapping)

    plt.figure(figsize=(18, 5))

    # (1) 原图
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Image')

    # (2) 叠加效果
    plt.subplot(1, 2, 2)
    plt.imshow(image)  # 先画底图
    plt.imshow(mask_rgb, alpha=0.6)  # 再叠加掩膜，alpha 决定透明度（0~1）
    plt.axis('off')
    plt.title('Mask')

    plt.tight_layout()
    plt.show()