from yacs.config import CfgNode as CN
from trainer import trainer
from networks.network import SwinUnet
from networks.network_CBAM import SwinUnet_CBAM
from networks.network_MCBAM import SwinUnet_MCBAM

def load_cfg(path="../configs/swinunet_mcbam_kvasir.yaml"):
    cfg = CN(new_allowed=True)      # 允许合并新字段
    cfg.merge_from_file(path)       # 把 YAML 合进来
    cfg.freeze()                    # 防止意外修改
    return cfg

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
    cfg = load_cfg(path="../configs/swinunet_kvasir.yaml")
    # network initialization
    #net = SwinUnet_MCBAM(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=cfg.MODEL.NUM_CLASSES).cuda()
    #net = SwinUnet_CBAM(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=cfg.MODEL.NUM_CLASSES).cuda()
    net = SwinUnet(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=cfg.MODEL.NUM_CLASSES).cuda()
    net.load_from(cfg)
    # dataset = 'Kvasir-SEG' 'CVC-ClinicDB' 'Synapse'
    trainer(config=cfg, dataset='Kvasir-SEG', model=net, snapshot_path="../output/SwinUnet_Kvasir",
            augmentation=True, dice_ratio=0.6)

    cfg = load_cfg(path="../configs/swinunet_cvc.yaml")
    net = SwinUnet(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=cfg.MODEL.NUM_CLASSES).cuda()
    net.load_from(cfg)
    trainer(config=cfg, dataset='CVC-ClinicDB', model=net, snapshot_path="../output/SwinUnet_CVC",
            augmentation=True, dice_ratio=0.6)

    cfg = load_cfg(path="../configs/swinunet_synapse.yaml")
    net = SwinUnet(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=cfg.MODEL.NUM_CLASSES).cuda()
    net.load_from(cfg)
    trainer(config=cfg, dataset='Synapse', model=net, snapshot_path="../output/SwinUnet_Synapse",
            augmentation=True, dice_ratio=0.6)

    cfg = load_cfg(path="../configs/swinunet_mcbam_synapse.yaml")
    net = SwinUnet_CBAM(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=cfg.MODEL.NUM_CLASSES).cuda()
    net.load_from(cfg)
    trainer(config=cfg, dataset='Synapse', model=net, snapshot_path="../output/SwinUnet_CBAM_Synapse",
            augmentation=True, dice_ratio=0.5)

    cfg = load_cfg(path="../configs/swinunet_mcbam_kvasir.yaml")
    net = SwinUnet_MCBAM(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=cfg.MODEL.NUM_CLASSES).cuda()
    net.load_from(cfg)
    trainer(config=cfg, dataset='Kvasir-SEG', model=net, snapshot_path="../output/SwinUnet_MCBAM_Kvasir",
            augmentation=True, dice_ratio=0.5)

    cfg = load_cfg(path="../configs/swinunet_mcbam_cvc.yaml")
    net = SwinUnet_MCBAM(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=cfg.MODEL.NUM_CLASSES).cuda()
    net.load_from(cfg)
    trainer(config=cfg, dataset='CVC-ClinicDB', model=net, snapshot_path="../output/SwinUnet_MCBAM_CVC",
            augmentation=True, dice_ratio=0.5)

    cfg = load_cfg(path="../configs/swinunet_mcbam_synapse.yaml")
    net = SwinUnet_MCBAM(cfg, img_size=cfg.DATA.IMG_SIZE, num_classes=cfg.MODEL.NUM_CLASSES).cuda()
    net.load_from(cfg)
    trainer(config=cfg, dataset='Synapse', model=net, snapshot_path="../output/SwinUnet_MCBAM_Synapse",
            augmentation=True, dice_ratio=0.5)