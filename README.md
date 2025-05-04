<!-- 项目徽章示例，可在 shields.io 生成 -->
![GitHub stars](https://img.shields.io/github/stars/yourname/yourrepo?style=social)
![License](https://img.shields.io/github/license/yourname/yourrepo)

# 🩺 MultiScale-Convolutional Block Attention Module SwinUNet
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

[📥 Download Swin-T Pre-trained Model] (https://1drv.ms/u/c/0e644bbccb1ebf6b/EXLv071dCmRKltuSPuMPgasBc11oauVDbNv7YLpT8EcBbA?e=cCGBYc)

Put pretrained Swin-T file 'swin_tiny_patch4_window7_224.pth' into folder 'configs/'

### 3. Download datasets
[Get Synapse, Kvasir-SEG, and CVC-ClinicDB datasets in this link]: 

- Link for Synapse (https://1drv.ms/u/c/0e644bbccb1ebf6b/EZhh3kY1WBNPo31PekMW-EAB2yTDtdJypReGMlbm7pVHBA?e=xfGPri)
- Link for Kvasir-SEG (https://1drv.ms/u/c/0e644bbccb1ebf6b/EWU0IBcfOeNPlHta9yUIo2AB9wGTeD4GyEyfT-zICsiUSw?e=XK1tXo)
- Link for CVC-ClinicDB (https://1drv.ms/u/c/0e644bbccb1ebf6b/EZ5io_b_efBFjyNQ--SrRLwB-OA3fLyWb8HChgYHwm1MNw?e=5t2gwk)

Unzip the datasets files into three folders, respectively.
- "datasets/Synapse"
- "datasets/Kvasir-SEG"
- "datasets/CVC-ClinicDB"  

Then run two .py scripts in the folder "datasets/" to split the data .

### 4. Configuration setting
In folder "./configs", .yaml documents save the configuration setting of each model for the specific dataset.

### 5. Train/Test
- Run the "./code/train.py" to run the model in the specified dataset.
- Remember in train.py, you need to specific which model (SwinUNet, CBAM SwinUNet, my proposed MS-CBAM SwinUNet) you want to run, 
and do not forget to set the address of configuration file .yaml, the dataset name.
- In terms of testing, set the configuration file, dataset name, the .pth file as well

### 6. Introduction of utils files 
They are both for the images visualization
- utilis1.py is used to look at the original image and check weather the image data can be data loaded properly.
- utilis1.py is used to look at the prediction on the test images.
