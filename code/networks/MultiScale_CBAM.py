import torch
from torch import nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // reduction, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, padding=3):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

# CBAM（Convolutional Block Attention Module）
class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        f1 = self.ca(x)
        #print(f"Shape of f1:{f1.shape}")
        out = x * f1
        #print(f"Shape of f':{out.shape}")
        f2 = self.sa(out)
        #print(f"Shape of f2:{f2.shape}")
        result = out *  f2
        #print(f"Shape of f'':{result.shape}")
        return result

# MultiScale-CBAM
class MultiScaleCBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, kernel_size=7):
        """
        参数:
          in_planes: 当前层特征的通道数，即最终输出的通道数 C

        注意:
          - 通道注意力分支输入特征尺寸为 (B, 2C, 0.5W, 0.5H)
          - 当前层特征尺寸为 (B, C, H, W)
          - 空间注意力分支输入特征尺寸为 (B, 0.5C, 2W, 2H)
        """
        super(MultiScaleCBAM, self).__init__()

        self.ca = ChannelAttention(in_planes * 2, reduction)
        self.conv_ca = nn.Conv2d(in_planes * 2, in_planes, kernel_size=1)

        self.sa = SpatialAttention(kernel_size)

    def forward(self, x_sa, x, x_ca):
        """
        参数:
          x: 当前层的特征，形状为 (B, C, H, W)
          x_ca: 用于通道注意力分支的多尺度特征，形状为 (B, 2C, 0.5W, 0.5H)
          x_sa: 用于空间注意力分支的多尺度特征，形状为 (B, 0.5C, 2W, 2H)
        """
        #print(f"Shape of shallow layer:{x_sa.shape}")
        #print(f"Shape of current layer:{x.shape}")
        #print(f"Shape of deep layer:{x_ca.shape}")
        # --------------- 通道注意力分支 ---------------
        attn_ca = self.ca(x_ca)
        attn_ca = self.conv_ca(attn_ca) # 通过1×1卷积调整通道数，从 2C 降为 C
        #print(f"Shape of channel factor:{attn_ca.shape}")
        out = x * attn_ca
        #print(f"Shape of out 1st:{out.shape}")

        # --------------- 空间注意力分支 ---------------
        # 得到注意力因子，形状为 (B, 1, 2W, 2H)
        attn_sa = self.sa(x_sa)
        #print(f"Shape of spatial factor:{attn_sa.shape}")
        # 下采样到原始的空间尺寸： (2W, 2H) -> (W, H)
        attn_sa = F.interpolate(attn_sa, scale_factor=0.5, mode='bilinear', align_corners=False)
        #print(f"Shape of spatial factor:{attn_sa.shape}")
        out = out * attn_sa
        #print(f"Shape of f2:{out.shape}")
        return out