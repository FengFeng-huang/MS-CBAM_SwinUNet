import os
import random
import numpy as np
import h5py
from PIL import Image
import shutil

# code for training dataset
def to_rgb(im2d):
    """把 (H, W) 灰度图扩展到 (H, W, 3)"""
    return np.stack([im2d, im2d, im2d], axis=-1)

def make_slices(src_dir, dst_dir, is_h5=False):
    """
    把 src_dir 下的 3D 数据切成 2D npz，丢弃全空 mask，保存到 dst_dir。
    同时将 image 从单通道扩展为三通道 RGB 格式 (H, W, 3)，mask 保持 (H, W)。
    is_h5=True 时处理 .npy.h5，False 时处理 .npz。
    返回所有生成的切片路径列表。
    """
    os.makedirs(dst_dir, exist_ok=True)
    out_paths = []

    if is_h5:
        files = [f for f in os.listdir(src_dir) if f.endswith(".npy.h5")]
        for fname in files:
            case = fname[:-len(".npy.h5")]
            with h5py.File(os.path.join(src_dir, fname), "r") as f:
                img_vol  = f["image"][:]
                mask_vol = f["label"][:]
            D = mask_vol.shape[2]
            for z in range(D):
                m = mask_vol[:, :, z]
                if not m.any():
                    continue
                im_rgb = to_rgb(img_vol[:, :, z])
                out_name = f"{case}_{z:03d}.npz"
                out_path = os.path.join(dst_dir, out_name)
                np.savez_compressed(out_path, image=im_rgb, label=m)
                out_paths.append(out_path)
    else:
        files = [f for f in os.listdir(src_dir) if f.endswith(".npz")]
        for fname in files:
            case = fname[:-4]
            data = np.load(os.path.join(src_dir, fname))
            img_vol  = data.get("image", data.get("data"))
            mask_vol = data.get("label", data.get("mask"))

            if mask_vol.ndim == 3:
                D = mask_vol.shape[2]
                for z in range(D):
                    m = mask_vol[:, :, z]
                    if not m.any():
                        continue
                    im_rgb = to_rgb(img_vol[:, :, z])
                    out_name = f"{case}_{z:03d}.npz"
                    out_path = os.path.join(dst_dir, out_name)
                    np.savez_compressed(out_path, image=im_rgb, label=m)
                    out_paths.append(out_path)
            else:
                m = mask_vol
                if not m.any():
                    continue
                im_rgb = to_rgb(img_vol)
                out_name = f"{case}.npz"
                out_path = os.path.join(dst_dir, out_name)
                np.savez_compressed(out_path, image=im_rgb, label=m)
                out_paths.append(out_path)

    return out_paths

def split_slices_to_png(slices_list, out_root, seed=42):
    """
    将切片列表全部划分为 train，
    并按 images/ (RGB) 和 masks/ (原值灰度) 子目录分别存为 PNG。

    返回字典：{'train': [...]}。
    """
    random.seed(seed)
    os.makedirs(out_root, exist_ok=True)
    # 创建 train 子目录结构
    train_dir = os.path.join(out_root, "train")
    for sub in ("images", "masks"):
        os.makedirs(os.path.join(train_dir, sub), exist_ok=True)

    shuffled = slices_list.copy()
    random.shuffle(shuffled)
    train_list = shuffled

    def _save_png(file_list, split):
        for path in file_list:
            data = np.load(path)
            im = data["image"]  # (H, W, 3)
            m  = data["label"]  # (H, W)

            base = os.path.splitext(os.path.basename(path))[0]

            # 保存 RGB 图像
            if im.dtype != np.uint8:
                im_uint8 = np.clip(im * 255, 0, 255).astype(np.uint8)
            else:
                im_uint8 = im
            Image.fromarray(im_uint8).save(
                os.path.join(out_root, split, "images", base + ".png")
            )

            # 保存原值灰度掩膜
            m_uint8 = m.astype(np.uint8)
            Image.fromarray(m_uint8, mode="L").save(
                os.path.join(out_root, split, "masks", base + ".png")
            )

    _save_png(train_list, "train")

    return {"train": train_list}

def process_slices(src_dir: str = "./Synapse/train_npz",slices_dir: str = "./Synapse/slices_2d",out_root: str = "./Synapse",
                   seed: int = 2025,is_h5: bool = False):
    # 1) 切片
    slices = make_slices(src_dir, slices_dir, is_h5=is_h5)
    print(f"切出 {len(slices)} 张 RGB+mask npz 切片。")

    # 2) 仅划分为训练集并保存为 PNG
    splits = split_slices_to_png(
        slices,
        out_root,
        seed=seed,
    )
    print(f"训练集: {len(splits['train'])} 张。")
    print("目录结构示例：")
    print(f"  {out_root}/train/images/")
    print(f"  {out_root}/train/masks/")

    return splits

# code for testing dataset
def dump_slices_to_temp(h5_folder, temp_folder):
    os.makedirs(temp_folder, exist_ok=True)
    for fname in os.listdir(h5_folder):
        if not fname.endswith('.npy.h5'):
            continue
        fpath = os.path.join(h5_folder, fname)
        with h5py.File(fpath, 'r') as f:
            # 根据实际 key 名称改这里
            mask_ds = f['mask'] if 'mask' in f else f[list(f.keys())[1]]
            img_ds  = f['image'] if 'image' in f else f[list(f.keys())[0]]
            depth = mask_ds.shape[0]

            base = os.path.splitext(fname)[0]
            for idx in range(depth):
                mask_slice = mask_ds[idx]
                img_slice  = img_ds[idx]

                # 灰度 or 单通道 扩到 3 通道
                if img_slice.ndim == 2:
                    img_slice = np.stack([img_slice]*3, axis=-1)
                elif img_slice.ndim == 3 and img_slice.shape[-1] == 1:
                    img_slice = np.concatenate([img_slice]*3, axis=-1)

                temp_name = f"{base}_{idx:03d}.npz"
                temp_path = os.path.join(temp_folder, temp_name)
                np.savez_compressed(temp_path,
                                    image=img_slice,
                                    label=mask_slice)
    print(f"已导出所有 slice 到：{temp_folder}")

def filter_zero_mask_slices(temp_folder, output_folder=None):
    """
    如果指定 output_folder，则会把非零 mask 的文件拷贝/移动到 output_folder；
    如果不指定，则直接在 temp_folder 内删除全零 mask 文件。
    """
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    for fname in os.listdir(temp_folder):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(temp_folder, fname)
        data = np.load(path)
        mask = data['label']
        data.close()

        if np.all(mask == 0):
            # 全零就删掉
            os.remove(path)
        else:
            # 如果要搬运到另一个文件夹：
            if output_folder:
                dst = os.path.join(output_folder, fname)
                os.replace(path, dst)
    print("筛选完成。")

def convert_npz_folder_to_png(npz_folder, out_root):
    """
    将 npz_folder 下所有 .npz 文件分别导出到 out_root/images 和 out_root/masks。
    假定每个 .npz 里：
      - image.shape == (512, 512, 3), dtype float 或 uint8
      - mask.shape  == (512, 512), dtype 任意
    """
    img_dir  = os.path.join(out_root, 'images')
    mask_dir = os.path.join(out_root, 'masks')
    os.makedirs(img_dir,  exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for fname in sorted(os.listdir(npz_folder)):
        if not fname.endswith('.npz'):
            continue

        path = os.path.join(npz_folder, fname)
        data = np.load(path)
        im = data['image']   # (512,512,3)
        m  = data['label']   # (512,512)
        data.close()

        base = os.path.splitext(fname)[0]

        # —— 保存 RGB 图像 —— #
        if im.dtype != np.uint8:
            im_uint8 = np.clip(im * 255, 0, 255).astype(np.uint8)
        else:
            im_uint8 = im
        Image.fromarray(im_uint8).save(os.path.join(img_dir,  base + '.png'))

        # —— 保存灰度掩膜 —— #
        # mask 已经是 (512,512)，直接转 uint8 即可
        m_uint8 = m.astype(np.uint8)
        Image.fromarray(m_uint8, mode='L').save(os.path.join(mask_dir, base + '.png'))

    print(f"Converted {len(os.listdir(img_dir))} images to {img_dir}")
    print(f"Converted {len(os.listdir(mask_dir))} masks  to {mask_dir}")

def split_png_dataset(png_folder, out_root, seed=2025):

    random.seed(seed)
    img_src  = os.path.join(png_folder, 'images')
    mask_src = os.path.join(png_folder, 'masks')

    # 1. 收集所有图像文件名（不带路径），并保证对应的 mask 存在
    all_imgs = [f for f in os.listdir(img_src) if f.lower().endswith('.png')]
    all_imgs = [f for f in all_imgs if os.path.exists(os.path.join(mask_src, f))]
    all_imgs.sort()

    # 2. 随机打乱并一分为二
    random.shuffle(all_imgs)
    half = len(all_imgs) // 2
    val_names  = all_imgs[:half]
    test_names = all_imgs[half:]

    # 3. 创建输出目录结构
    for split in ('val', 'test'):
        for sub in ('images', 'masks'):
            os.makedirs(os.path.join(out_root, split, sub), exist_ok=True)

    # 4. 复制文件
    for split, names in (('val', val_names), ('test', test_names)):
        for name in names:
            shutil.copy2(os.path.join(img_src,  name),
                         os.path.join(out_root, split, 'images', name))
            shutil.copy2(os.path.join(mask_src, name),
                         os.path.join(out_root, split, 'masks',  name))

    return {'val': val_names, 'test': test_names}

def main():
    # training dataset allocated
    training_splits = process_slices(src_dir="./Synapse/train_npz", slices_dir="./Synapse/slices_2d", out_root="./Synapse",
                            seed=2025, is_h5=False)
    # testing dataset allocated
    # 第一步：导出所有切片
    dump_slices_to_temp('./Synapse/test_vol_h5', './Synapse/temp_slices')
    # 第二步：删除全零 mask 或者搬运非零到 final
    filter_zero_mask_slices('./Synapse/temp_slices', './Synapse/final_slices')
    # 第三步，最终创建数据集
    convert_npz_folder_to_png('./Synapse/final_slices', './Synapse/png_export')
    # 最后一步，二等分测试集
    testing_splits = split_png_dataset('./Synapse/png_export', './Synapse')
    print(f"Val 集合:  {len(testing_splits['val'])} 张")
    print(f"Test 集合: {len(testing_splits['test'])} 张")
    print("Synapse dataset splitting is done!")

if __name__ == "__main__":
    main()