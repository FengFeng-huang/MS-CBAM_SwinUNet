import os
from dataloader import ImageMaskDataset
import torch
from torch.utils.data import  DataLoader
import torch.nn as nn
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm
from loss_function import DiceLoss

def trainer(config, dataset, model, snapshot_path, augmentation=False, dice_ratio=0.6):
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    print("Configuration:", config)
    print("Images Augmentation is:", augmentation)

    base_lr = config.TRAIN.BASE_LR
    num_classes = config.MODEL.NUM_CLASSES
    batch_size = config.DATA.BATCH_SIZE * config.TRAIN.NUM_GPUS
    base_momentum = config.TRAIN.OPTIMIZER.MOMENTUM
    base_weight_decay = config.TRAIN.WEIGHT_DECAY

    db_train = ImageMaskDataset(
        os.path.join('..', 'datasets', dataset, 'train', 'images'),
        os.path.join('..', 'datasets', dataset, 'train', 'masks'),
        dataset, transform_image=None, transform_mask=None,
        augment=augmentation
    )
    db_val = ImageMaskDataset(
        os.path.join('..', 'datasets', dataset, 'val', 'images'),
        os.path.join('..', 'datasets', dataset, 'val', 'masks'),
        dataset, transform_image=None, transform_mask=None,
    )
    print("The length of train set is: {}".format(len(db_train)))
    print("The length of validation set is: {}".format(len(db_val)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    if config.TRAIN.NUM_GPUS > 1:
        model = nn.DataParallel(model)

    model.train()

    # Loss function
    ce_loss = nn.CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)

    # Optimizer
    if config.TRAIN.OPTIMIZER.NAME.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=config.TRAIN.OPTIMIZER.BETAS,
            eps=config.TRAIN.OPTIMIZER.EPS,
            weight_decay=base_weight_decay
        )
    elif config.TRAIN.OPTIMIZER.NAME.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=base_lr,
            momentum=base_momentum,
            weight_decay=base_weight_decay
        )

    # lr scheduler
    if config.TRAIN.LR_SCHEDULER.NAME.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAIN.EPOCHS)
    elif config.TRAIN.LR_SCHEDULER.NAME.lower() == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS,
                                                    gamma=config.TRAIN.LR_SCHEDULER.DECAY_RATE
                                                    )
    else:
        scheduler = None

    writer = SummaryWriter(snapshot_path + '/log')

    # each epoch training and validation loss
    train_loss_history = []
    val_loss_history = []

    # setting for training loop
    iter_num = 0
    max_epoch = config.TRAIN.EPOCHS
    max_iterations = config.TRAIN.EPOCHS * len(trainloader)
    print("{} iterations per epoch. {} max iterations".format(len(trainloader), max_iterations))

    epoch_iterator = tqdm(range(max_epoch), ncols=70)

    # initialize the best loss
    best_val_loss = float('inf')
    best_model_path = os.path.join(snapshot_path, 'best_model.pth')

    # Training loop
    for epoch_num in epoch_iterator:

        model.train()
        running_loss = 0.0
        num_batches = 0

        for i_batch, (image, mask) in enumerate(trainloader):
            image_batch, label_batch = image, mask
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()

            outputs = model(image_batch)  # (B,C,H,W)

            # Training Loss
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = (1 - dice_ratio) * loss_ce + dice_ratio * loss_dice

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1
            running_loss += loss.item()
            num_batches += 1

        if scheduler is not None:
            scheduler.step()

        # 计算本 epoch 的平均训练 loss
        avg_train_loss = running_loss / num_batches
        train_loss_history.append(avg_train_loss)
        writer.add_scalar('epoch/train_loss', avg_train_loss, epoch_num)
        print("Epoch {}: Average training loss: {}".format(epoch_num, avg_train_loss))

        # validation
        model.eval()
        running_val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for image, mask in valloader:
                image = image.cuda()
                mask = mask.cuda()
                outputs = model(image)
                loss_ce = ce_loss(outputs, mask.long())
                loss_dice = dice_loss(outputs, mask, softmax=True)
                loss_val = (1 - dice_ratio) * loss_ce + dice_ratio * loss_dice
                running_val_loss += loss_val.item()
                val_batches += 1
        avg_val_loss = running_val_loss / val_batches
        val_loss_history.append(avg_val_loss)
        writer.add_scalar('epoch/val_loss', avg_val_loss, epoch_num)
        print("Epoch {}: Average validation loss: {}".format(epoch_num, avg_val_loss))

        # 保存最佳模型：当本 epoch 验证 loss 更低时覆盖保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model, best_model_path)
            print("Epoch {}: Best model saved, validation loss: {}".format(epoch_num, avg_val_loss))

    # 保存最后一次训练模型
    final_model_path = os.path.join(snapshot_path, 'final_model.pth')
    torch.save(model, final_model_path)
    print("Final model saved to", final_model_path)

    writer.close()

    # 绘制训练和验证 loss 曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(max_epoch), train_loss_history, label='Train Loss')
    plt.plot(range(max_epoch), val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    best_epoch_index = val_loss_history.index(min(val_loss_history))
    best_loss = min(val_loss_history)
    plt.scatter(best_epoch_index, best_loss, color='red', marker='o', s=100, label='Best Model')
    plt.axhline(y=best_loss, linestyle='--', color='gray')
    plt.axvline(x=best_epoch_index, linestyle='--', color='gray')
    plt.text(best_epoch_index, best_loss, f'\nEpoch: {best_epoch_index}\nLoss: {best_loss:.4f}',
             fontsize=10, color='black', verticalalignment='bottom', horizontalalignment='left')
    loss_plot_path = os.path.join(snapshot_path, 'loss_curve.png')
    plt.savefig(loss_plot_path)
    plt.show()

    return "Training Finished!"