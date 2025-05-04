<!-- 项目徽章示例，可在 shields.io 生成 -->
![GitHub stars](https://img.shields.io/github/stars/yourname/yourrepo?style=social)
![License](https://img.shields.io/github/license/yourname/yourrepo)

# MultiScale-Convolutional Block Attention Module SwinUNet
> *“This is my Final Year Project in XJTLU undergraduate.”*  

---

## 📑 目录 | Table of Contents
1. [QuickStart](#-快速开始-quick-start)  
2. [Environment](#-环境要求-environment)  
3. [Pre‑trained Model](#-预训练模型下载-pre-trained-model)  
4. [Datasets](#-数据集下载-datasets)  
5. [Config Files](#-配置文件-config-files)  
6. [Train &Test](#-训练--测试-train--test)  
7. [Utils](#-可视化工具-utils)

---

## 🚀 Quick Start

### 1. Environment requirements
Python == 3.8
torch == 2.4.1
torchvision == 0.19.1

### 2. Download pre-trained swin transformer model (Swin-T)
Download the pre-trained model from the link below:

[📥 Download Swin-T Pre-trained Model](https://1drv.ms/u/c/0e644bbccb1ebf6b/EXLv071dCmRKltuSPuMPgasBc11oauVDbNv7YLpT8EcBbA?e=cCGBYc)

Put pretrained Swin-T file 'swin_tiny_patch4_window7_224.pth' into folder 'configs/'

### 3. Download datasets
- [📥 Synapse Dataset](https://1drv.ms/u/c/0e644bbccb1ebf6b/EZhh3kY1WBNPo31PekMW-EAB2yTDtdJypReGMlbm7pVHBA?e=xfGPri)
- [📥 Kvasir-SEG Dataset](https://1drv.ms/u/c/0e644bbccb1ebf6b/EWU0IBcfOeNPlHta9yUIo2AB9wGTeD4GyEyfT-zICsiUSw?e=XK1tXo)
- [📥 CVC-ClinicDB Dataset](https://1drv.ms/u/c/0e644bbccb1ebf6b/EZ5io_b_efBFjyNQ--SrRLwB-OA3fLyWb8HChgYHwm1MNw?e=5t2gwk)

After downloading, unzip each dataset into the following folders:
- 'datasets/Synapse'
- 'datasets/Kvasir-SEG'
- 'datasets/CVC-ClinicDB'

Then, run the two Python scripts located in the `datasets/` folder to split the data appropriately.

### 4. Configuration setting
All configuration files are stored in the `./configs` folder.  
Each `.yaml` file contains the specific settings for a given model and dataset combination.

### 5. Train / Test
#### 🏋️‍♂️ Training

To train a model, run the following script:

```bash
python code/train.py
```
Make sure to specify:
- The model you want to use:
  - SwinUNet, 
  - CBAM_SwinUNet
  - my proposed MS-CBAM SwinUNet
- The path to the corresponding .yaml configuration file
- The dataset name

#### 🧪 Testing
To test a model, run the following script:
```bash
python code/test.py
```
To evaluate a trained model, configure the following:
- The path to the same `.yaml` config file used for training
- The dataset name
- The path to the saved `.pth` checkpoint file

### 6. Introduction of utils files 
These Python scripts are used for image visualization:
- `utilis1.py`: Visualizes the original images to check whether the image data can be loaded correctly.
- `utilis2.py`: Visualizes the model's predictions on test images.
