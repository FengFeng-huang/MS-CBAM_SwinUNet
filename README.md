<!-- 项目徽章示例，可在 shields.io 生成 -->
![GitHub stars](https://img.shields.io/github/stars/yourname/yourrepo?style=social)
![License](https://img.shields.io/github/license/yourname/yourrepo)

# 项目名称 Project Name
一句话 / One-liner tagline

## 🚀 快速开始 Quick Start

# 1. Download pre-trained swin transformer model (Swin-T)
[Get pre-trained model in this link]
(https://1drv.ms/u/c/0e644bbccb1ebf6b/EXLv071dCmRKltuSPuMPgasBc11oauVDbNv7YLpT8EcBbA?e=nHSpGf): Put pretrained Swin-T file 'swin_tiny_patch4_window7_224.pth' into folder "configs/"

# 2. Download datasets
[[Get Synapse, Kvasir-SEG, and CVC-ClinicDB datasets in this link]
Link for Synapse (https://1drv.ms/u/c/0e644bbccb1ebf6b/EZhh3kY1WBNPo31PekMW-EAB2yTDtdJypReGMlbm7pVHBA?e=xfGPri)
Link for Kvasie-SEG (https://1drv.ms/u/c/0e644bbccb1ebf6b/EWU0IBcfOeNPlHta9yUIo2AB9wGTeD4GyEyfT-zICsiUSw?e=6mqE8Z)
Link for CVC-ClinicDB (https://1drv.ms/u/c/0e644bbccb1ebf6b/EZ5io_b_efBFjyNQ--SrRLwB-OA3fLyWb8HChgYHwm1MNw?e=l0KMbC)

Unzip the datasets files into folders "datasets/Synapse", "datasets/Kvasir-SEG", "datasets/CVC-ClinicDB",  respectively.
Then run two .py scripts in the folder "datasets/" to realize the data spliting.

# 安装依赖
npm install    # 或 pip install -r requirements.txt

# 本地运行
npm run dev    # or python main.py
