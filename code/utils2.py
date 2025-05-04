from train import load_cfg
from dataloader import ImageMaskDataset
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

def evaluate_segmentation(config, dataset_name, split, model_path):
    """
    Evaluate a segmentation model over a dataset split.

    Computes per-class and mean IoU and Dice (excluding background label 0),
    plus macro-precision, recall, F1. Returns metrics and raw predictions for
    downstream visualization.

    Returns:
        metrics: dict with keys:
            - per_class_iou: list of IoUs for classes 1..C-1
            - per_class_dice: list of DSCs for classes 1..C-1
            - mean_iou: float
            - mean_dice: float
            - precision_macro, recall_macro, f1_macro: floats
        samples: dict with keys:
            - images: list of np.ndarray sample images
            - preds: list of 2D np.ndarray predicted labels
            - labels: list of 2D np.ndarray ground truth labels
            - sample_ious: list of mean IoU per sample (classes 1..C-1)
            - sample_dices: list of mean DSC per sample (classes 1..C-1)
    """
    # --- 1. Prepare DataLoader ---
    test_image_dir = os.path.join('..', 'datasets', dataset_name, split, 'images')
    test_mask_dir  = os.path.join('..', 'datasets', dataset_name, split, 'masks')
    test_ds = ImageMaskDataset(
        test_image_dir,
        test_mask_dir,
        dataset_name=dataset_name,
        transform_image=None,
        transform_mask=None,
        size=(224, 224)
    )
    loader = DataLoader(
        test_ds,
        batch_size=config.DATA.BATCH_SIZE * config.TRAIN.NUM_GPUS,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # --- 2. Load model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.to(device).eval()

    # Collectors
    all_preds, all_labels = [], []
    sample_images, sample_ious, sample_dices = [], [], []
    C = config.MODEL.NUM_CLASSES

    # --- 3. Inference & per-sample metrics ---
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            for b in range(preds.size(0)):
                p = preds[b].cpu().numpy()
                l = masks[b].cpu().numpy()
                all_preds.append(p)
                all_labels.append(l)
                # per-sample metrics (exclude class 0)
                ious, dices = [], []
                for cls in range(1, C):
                    inter = np.logical_and(p==cls, l==cls).sum()
                    uni   = np.logical_or(p==cls, l==cls).sum()
                    ious.append(1.0 if uni==0 else inter/uni)
                    pred_sum, label_sum = (p==cls).sum(), (l==cls).sum()
                    dices.append(1.0 if (pred_sum+label_sum)==0 else 2*inter/(pred_sum+label_sum))
                sample_ious.append(np.mean(ious))
                sample_dices.append(np.mean(dices))
                # save image for visualization
                img_np = images[b].cpu().numpy()
                if img_np.ndim == 3 and img_np.shape[0] > 1:
                    img_np = np.transpose(img_np, (1,2,0))
                sample_images.append(img_np)

    # --- 4. Global metrics ---
    y_pred = np.concatenate([p.flatten() for p in all_preds])
    y_true = np.concatenate([l.flatten() for l in all_labels])
    ious_per_class, dices_per_class = [], []
    for cls in range(1, C):
        inter = np.logical_and(y_true==cls, y_pred==cls).sum()
        uni   = np.logical_or(y_true==cls, y_pred==cls).sum()
        ious_per_class.append(1.0 if uni==0 else inter/uni)
        pred_sum, label_sum = (y_pred==cls).sum(), (y_true==cls).sum()
        dices_per_class.append(1.0 if (pred_sum+label_sum)==0 else 2*inter/(pred_sum+label_sum))
    mean_iou = float(np.mean(ious_per_class))
    mean_dice = float(np.mean(dices_per_class))
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score   (y_true, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score       (y_true, y_pred, average='macro', zero_division=0)

    metrics = {
        'per_class_iou': ious_per_class,
        'per_class_dice': dices_per_class,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }
    samples = {
        'images': sample_images,
        'preds': all_preds,
        'labels': all_labels,
        'sample_ious': sample_ious,
        'sample_dices': sample_dices
    }
    return metrics, samples

def mosaic_sample(sample_idx, samples, save_path=None):
    """
    Display two overlays for a specific sample across all non-background classes using custom colors.

    samples: dict returned by evaluate_segmentation
    sample_idx: index of the sample in samples['images']
    save_path: if provided, save the figure to this path
    """
    img = samples['images'][sample_idx]
    gt = samples['labels'][sample_idx]
    pred = samples['preds'][sample_idx]

    # prepare display image
    if img.dtype != np.uint8:
        img_disp = (img * 255).astype(np.uint8) if img.max() <= 1 else img.astype(np.uint8)
    else:
        img_disp = img

    # 构建彩色掩码图
    h, w = gt.shape
    gt_color = np.zeros((h, w, 3), dtype=np.uint8)
    pred_color = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_mapping.items():
        if label == 0:
            continue
        gt_color[gt == label] = color
        pred_color[pred == label] = color

    # 读取样本指标
    iou = samples['sample_ious'][sample_idx]
    dice = samples['sample_dices'][sample_idx]

    # Plot overlays
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    titles = ["GT Overlay", f"Pred Overlay\nIoU={iou:.4f}, Dice={dice:.4f}"]
    overlays = [gt_color, pred_color]

    for ax, overlay, title in zip(axes, overlays, titles):
        ax.imshow(img_disp)
        ax.imshow(overlay, alpha=0.5)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Overlay figure saved to: {save_path}")
    plt.show()

if __name__ == '__main__':
    # color map：label -> RGB
    color_mapping = {
        1: (255, 255, 255),  # 白色  spleen
        2: (255, 182, 193),  # 浅粉红 right kidney
        3: (0, 0, 255),      # 红色  left kidney
        4: (0, 255, 0),      # 绿色  gallbladder
        5: (255, 0, 0),      # 蓝色  liver
        6: (0, 255, 255),    # 黄色  stomach
        7: (255, 0, 255),    # 紫色  aorta
        8: (255, 255, 0)     # 青色  pancreas
    }
    # configuration loading
    '''
    swinunet_mcbam_synapse.yaml
    swinunet_mcbam_kvasir.yaml
    swinunet_mcbam_cvc.yaml

    swinunet_synapse.yaml
    swinunet_kvasir.yaml
    swinunet_cvc.yaml
    '''
    idx_num=71
    cfg = load_cfg(path="../configs/swinunet_mcbam_synapse.yaml")
    metrics, samples = evaluate_segmentation(cfg, 'Synapse', 'test', '../output/SwinUnet_MCBAM_Synapse/best_model.pth')
    mosaic_sample(idx_num, samples, save_path='../figures/sample5_overlay.png')
    cfg = load_cfg(path="../configs/swinunet_synapse.yaml")
    metrics, samples = evaluate_segmentation(cfg, 'Synapse', 'test', '../output/SwinUnet_Synapse/best_model.pth')
    mosaic_sample(idx_num, samples, save_path='../figures/sample5_overlay.png')