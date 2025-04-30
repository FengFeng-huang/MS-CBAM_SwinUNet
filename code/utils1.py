import numpy as np
from torch.utils.data import DataLoader
from dataloader import ImageMaskDataset, check_dataloader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Check images in the dataset
    synapse_dataset = ImageMaskDataset(image_dir="../datasets/Synapse/train/images", mask_dir="../datasets/Synapse/train/masks",
                                       dataset_name='Synapse', augment=True)
    synapse_train_loader = DataLoader(synapse_dataset, batch_size=1, shuffle=True)

    print("Train Loader:")
    check_dataloader(synapse_train_loader)

    # Show the image
    image, mask = next(iter(synapse_train_loader))
    image = image[0].permute(1, 2, 0).numpy()
    mask = mask[0].numpy()

    print(np.unique(mask))

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Image')
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.axis('off')
    plt.title('Mask')
    plt.show()