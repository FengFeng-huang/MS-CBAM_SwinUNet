import numpy as np
import os
import matplotlib.pyplot as plt
from dataloader import ImageMaskDataset
import torch
from torch.utils.data import DataLoader
from train import load_cfg
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
from medpy.metric.binary import hd95 as compute_hd95

def evaluate_model(config, dataset_name, split, model_path):
    """
    config: 配置对象，包含 DATA.BATCH_SIZE, TRAIN.NUM_GPUS, MODEL.NUM_CLASSES 等
    dataset_name: 字符串，例如 'Synapse' 或其他
    split:      字符串，'train' / 'val' / 'test' 等
    model_path: 模型文件路径
    """

    # 1. 构造 DataLoader
    img_save_dir = os.path.dirname(model_path)
    test_image_dir = os.path.join('..', 'datasets', dataset_name, split, 'images')
    test_mask_dir  = os.path.join('..', 'datasets', dataset_name, split, 'masks')
    test_ds = ImageMaskDataset(test_image_dir,test_mask_dir,dataset_name=dataset_name,
        transform_image=None,
        transform_mask=None,
        size=(224, 224),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size= config.DATA.BATCH_SIZE * config.TRAIN.NUM_GPUS,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 2. 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.to(device).eval()

    all_preds = []
    all_labels = []
    sample_images = []
    sample_ious = []

    # 3. 推理并收集
    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)   # [B, 3, H, W]
            masks  = masks.to(device)    # [B, H, W]

            outputs = model(images)             # [B, C, H, W]
            preds   = outputs.argmax(dim=1)     # [B, H, W]

            for b in range(preds.size(0)):
                p = preds[b].cpu().numpy()
                l = masks[b].cpu().numpy()
                all_preds.append(p)
                all_labels.append(l)

                # per-sample IoU
                ious = []
                for cls in range(config.MODEL.NUM_CLASSES):
                    inter = np.logical_and(p==cls, l==cls).sum()
                    uni   = np.logical_or(p==cls, l==cls).sum()
                    ious.append(1.0 if uni==0 else inter/uni)
                sample_ious.append(np.mean(ious))

                # 保存原图用于可视化
                img_np = images[b].cpu().numpy()
                if img_np.shape[0] == 1:
                    img_np = img_np[0]
                else:
                    img_np = np.transpose(img_np, (1,2,0))
                sample_images.append(img_np)

    # 4. Flatten 全局指标
    y_pred = np.concatenate([p.flatten() for p in all_preds])
    y_true = np.concatenate([l.flatten() for l in all_labels])
    C = config.MODEL.NUM_CLASSES

    # 混淆矩阵 & 全局 IoU
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    ious_per_class = []
    for cls in range(C):
        inter = np.logical_and(y_true==cls, y_pred==cls).sum()
        uni   = np.logical_or(y_true==cls, y_pred==cls).sum()
        ious_per_class.append(1.0 if uni==0 else inter/uni)
    mean_iou = np.mean(ious_per_class)

    # 多分类 Precision/Recall/F1 (macro)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro    = recall_score   (y_true, y_pred, average='macro', zero_division=0)
    f1_macro        = f1_score       (y_true, y_pred, average='macro', zero_division=0)

    # 5. 各类别 HD95
    hd95_per_class = []
    for cls in range(C):
        cls_hd95 = []
        for p, l in zip(all_preds, all_labels):
            bin_p = (p == cls)
            bin_l = (l == cls)
            if not bin_p.any() and not bin_l.any():
                cls_hd95.append(0.0)
            elif not bin_p.any() or not bin_l.any():
                cls_hd95.append(np.nan)
            else:
                cls_hd95.append(compute_hd95(bin_l, bin_p))
        hd95_per_class.append(np.nanmean(cls_hd95))

    # Dice Coefficient Score
    dsc_per_class = []
    for cls in range(C):
        inter    = np.logical_and(y_true==cls, y_pred==cls).sum()
        pred_sum = (y_pred==cls).sum()
        label_sum= (y_true==cls).sum()
        if pred_sum + label_sum == 0:
            dsc = 1.0  # 如果该类在所有样本中都没出现，定义 DSC=1
        else:
            dsc = 2 * inter / (pred_sum + label_sum)
        dsc_per_class.append(dsc)

    # 6. 打印指标
    # Including label 0 (background)
    mean_dsc = np.mean(dsc_per_class)
    mean_iou = np.mean(ious_per_class)
    # Excluding label 0 (background)
    mean_iou_excl_bg = np.mean(ious_per_class[1:])
    mean_dsc_excl_bg = np.mean(dsc_per_class[1:])
    print("==== Global Metrics ====")
    print(f"Mean Precision (macro): {precision_macro:.4f}")
    print(f"Mean Recall    (macro): {recall_macro:.4f}")
    print(f"Mean F1-score  (macro): {f1_macro:.4f}")
    print(f"Mean IoU                : {mean_iou:.4f}")
    print("Mean HD95 per class (mm):")
    for cls, hd in enumerate(hd95_per_class):
        print(f"  Class {cls}: {hd:.4f}")
    print()

    print("==== Average over Classes ====")
    print(f"  Avg IoU (incl. background):    {mean_iou:.4f}")
    print(f"  Avg DSC (incl. background):    {mean_dsc:.4f}")
    print(f"  Avg IoU (excl. background):    {mean_iou_excl_bg:.4f}")
    print(f"  Avg DSC (excl. background):    {mean_dsc_excl_bg:.4f}")
    print()

    print("==== Per-class IoU & DSC ====")
    for cls in range(C):
        print(f"  Class {cls}: IoU = {ious_per_class[cls]:.4f}    DSC = {dsc_per_class[cls]:.4f}")
    print()

    # 7. 绘制混淆矩阵
    plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(C)
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    thresh = cm.max() / 2.
    for i in range(C):
        for j in range(C):
            plt.text(j, i, cm[i, j], ha='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = os.path.join(os.getcwd(), "confusion_matrix.png")
    plt.savefig(os.path.join(img_save_dir,cm_path))
    plt.show()
    print("Confusion matrix saved to:", os.path.join(img_save_dir,cm_path))

    # 8. 可视化 IoU 最好 / 最差 样本
    def plot_samples(indices, title, save_path):
        fig, axes = plt.subplots(len(indices), 3, figsize=(15, 5*len(indices)))
        fig.suptitle(title, fontsize=16)
        for row, idx in enumerate(indices):
            img = sample_images[idx]
            if img.dtype != np.uint8 and img.max() <= 1:
                img_disp = (img*255).astype(np.uint8)
            else:
                img_disp = img
            axes[row,0].imshow(img_disp)
            axes[row,0].set_title("Original");    axes[row,0].axis('off')

            axes[row,1].imshow(all_labels[idx], cmap='gray')
            axes[row,1].set_title("Ground Truth"); axes[row,1].axis('off')

            axes[row,2].imshow(all_preds[idx], cmap='gray')
            axes[row,2].set_title(f"Prediction\nIoU: {sample_ious[idx]:.4f}")
            axes[row,2].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(save_path)
        plt.show()
        print(f"{title} saved to: {save_path}")

    sample_ious_arr = np.array(sample_ious)
    top3 = sample_ious_arr.argsort()[-3:][::-1]
    bot3 = sample_ious_arr.argsort()[:3]
    plot_samples(top3, "Top 3 Highest IoU Samples", os.path.join(img_save_dir, "high_iou_samples.png"))
    plot_samples(bot3, "Top 3 Lowest IoU Samples",  os.path.join(img_save_dir, "low_iou_samples.png"))

    # 9. 返回结果
    print(f"The evaluation results is save to: {img_save_dir}")
    return

if __name__ == "__main__":
    # configuration loading
    '''
    swinunet_mcbam_synapse.yaml
    swinunet_mcbam_kvasir.yaml
    swinunet_mcbam_cvc.yaml

    swinunet_synapse.yaml
    swinunet_kvasir.yaml
    swinunet_cvc.yaml
    '''
    cfg = load_cfg(path="../configs/swinunet_mcbam_synapse.yaml")
    # inferencing and evaluating on test dataset
    evaluate_model(config=cfg, dataset_name='Synapse', split='test',
                   model_path='../output/SwinUnet_MCBAM_Synapse/best_model.pth')