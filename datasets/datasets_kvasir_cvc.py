import os
import random
import shutil

def split_dataset(
        image_dir: str,mask_dir: str,dest_root: str = '.',
        train_ratio: float = 0.7,val_ratio: float = 0.15,seed: int = 42
):

    # 获取所有图像文件名
    images = [f for f in os.listdir(image_dir)
              if f.lower().endswith(('.jpg', '.tif', '.tiff'))]

    # 打乱并划分索引
    random.seed(seed)
    random.shuffle(images)
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # 定义目标目录结构
    dest_dirs = {
        'train': {
            'images': os.path.join(dest_root, 'train', 'images'),
            'masks': os.path.join(dest_root, 'train', 'masks')
        },
        'val': {
            'images': os.path.join(dest_root, 'val', 'images'),
            'masks': os.path.join(dest_root, 'val', 'masks')
        },
        'test': {
            'images': os.path.join(dest_root, 'test', 'images'),
            'masks': os.path.join(dest_root, 'test', 'masks')
        }
    }
    # 创建目录
    for split in dest_dirs.values():
        os.makedirs(split['images'], exist_ok=True)
        os.makedirs(split['masks'], exist_ok=True)

    # 复制文件的辅助函数
    def _copy_files(file_list, dst_img_dir, dst_mask_dir):
        for fname in file_list:
            src_img = os.path.join(image_dir, fname)
            src_mask = os.path.join(mask_dir, fname)
            if not os.path.exists(src_mask):
                print(f"Warning: mask file not found for {fname}")
                continue
            shutil.copy(src_img, os.path.join(dst_img_dir, fname))
            shutil.copy(src_mask, os.path.join(dst_mask_dir, fname))

    # 执行复制
    _copy_files(train_files, dest_dirs['train']['images'], dest_dirs['train']['masks'])
    _copy_files(val_files,   dest_dirs['val']['images'],   dest_dirs['val']['masks'])
    _copy_files(test_files,  dest_dirs['test']['images'],  dest_dirs['test']['masks'])

    # 输出统计信息
    print(f"Total images: {total}")
    print(f"Train images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Test images: {len(test_files)}")
    print("数据集分割完成！")

    return {'train': train_files, 'val': val_files, 'test': test_files}

if __name__ == "__main__":
    # splitting with the 70 : 15: 15
    # splitting CVC-ClinicDB
    splits = split_dataset(image_dir="./CVC-ClinicDB/Original", mask_dir="./CVC-ClinicDB/Ground Truth",
                           dest_root='./CVC-ClinicDB', seed=2025)
    # splitting Kvasir-SEG
    splits = split_dataset(image_dir="./Kvasir-SEG/images", mask_dir="./Kvasir-SEG/masks",
                           dest_root='./Kvasir-SEG/', seed=2025)