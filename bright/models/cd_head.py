"""
UperNet 变化检测头 — PPM + FPN 架构。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PPM(nn.Module):
    """Pyramid Pooling Module — 多尺度上下文聚合

    参考 mmsegmentation 实现:
    - 多个池化尺度捕获多尺度上下文
    - 瓶颈层将拼接特征压缩回 channels
    """

    def __init__(self, pool_scales, in_channels, channels):
        super().__init__()
        self.pool_scales = pool_scales
        self.scale_convs = nn.ModuleList()
        for scale in pool_scales:
            self.scale_convs.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(scale),
                nn.Conv2d(in_channels, channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))
        # 拼接后: in_channels + channels * len(pool_scales) → channels
        cat_channels = in_channels + channels * len(pool_scales)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(cat_channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = [x]
        for conv in self.scale_convs:
            pooled = conv(x)
            out.append(F.interpolate(
                pooled, size=x.shape[-2:], mode='bilinear', align_corners=False
            ))
        out = torch.cat(out, dim=1)
        return self.bottleneck(out)


class UPerHead(nn.Module):
    """
    UperNet 解码头 — 参考 mmsegmentation 实现。

    架构:
    1. PPM 作用在最深层特征
    2. FPN top-down + lateral connections (每层有融合卷积)
    3. 所有尺度上采样拼接后分类

    输入: DINOv3/FusionViT 输出的4层多尺度特征
    输出: 二分类变化图 [B, 2, H, W]
    """

    def __init__(self,
                 in_channels=[768, 768, 768, 768],  # 4层特征通道 (ViT-B)
                 channels=512,                        # 中间通道
                 pool_scales=(1, 2, 3, 6),            # PPM 池化尺度
                 num_classes=2,                       # 变化/不变
                 dropout_ratio=0.1,
                 align_corners=False):
        super().__init__()
        self.align_corners = align_corners
        self.num_levels = len(in_channels)

        # PPM 模块 (作用在最深层)
        self.ppm = PPM(pool_scales, in_channels[-1], channels)

        # FPN 侧边连接 (1×1 conv 对齐通道)
        self.lateral_convs = nn.ModuleList()
        for i in range(self.num_levels):
            self.lateral_convs.append(nn.Sequential(
                nn.Conv2d(in_channels[i], channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))

        # FPN 融合卷积 (lateral + 上采样后融合)
        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_levels):
            self.fpn_convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))

        # FPN 拼接后的瓶颈层
        self.fpn_bottleneck = nn.Sequential(
            nn.Conv2d(channels * self.num_levels, channels,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # 最终分类头
        self.cls_seg = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_ratio),
            nn.Conv2d(channels, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        """
        Args:
            inputs: list of 4 tensors [B, C_i, H_i, W_i]
                    从浅到深排列 (inputs[0] 最浅/最大分辨率)
        Returns:
            seg_logits: [B, num_classes, H, W]
        """
        assert len(inputs) == self.num_levels, \
            f"Expected {self.num_levels} inputs, got {len(inputs)}"

        # PPM on deepest layer
        ppm_out = self.ppm(inputs[-1])  # [B, channels, H_deep, W_deep]

        # Lateral convs
        laterals = []
        for i, feat in enumerate(inputs):
            laterals.append(self.lateral_convs[i](feat))

        # FPN top-down: 从深到浅
        # ppm_out 在最深层与 lateral[-1] 融合
        f = laterals[-1] + F.interpolate(
            ppm_out, size=laterals[-1].shape[-2:], mode='bilinear',
            align_corners=self.align_corners
        )
        f_out = self.fpn_convs[-1](f)

        fused_outputs = [None] * self.num_levels
        fused_outputs[-1] = f_out

        # 逐层向上融合
        for i in range(self.num_levels - 2, -1, -1):
            f = laterals[i] + F.interpolate(
                fused_outputs[i + 1],
                size=laterals[i].shape[-2:],
                mode='bilinear',
                align_corners=self.align_corners
            )
            fused_outputs[i] = self.fpn_convs[i](f)

        # 所有尺度上采样到最浅层分辨率后拼接
        h_shallow = inputs[0].shape[-2:]
        f_cat = [
            fused_outputs[0],
        ]
        for i in range(1, self.num_levels):
            f_cat.append(F.interpolate(
                fused_outputs[i], size=h_shallow,
                mode='bilinear', align_corners=self.align_corners
            ))
        f_cat = torch.cat(f_cat, dim=1)

        f_fused = self.fpn_bottleneck(f_cat)

        # 最终预测
        out = self.cls_seg(f_fused)
        # 上采样到原图尺寸: 输入 224×224, 特征图 16×16 (224/14)
        # scale_factor = 14 回到 224×224
        # 但 ViT-B/14 输出固定 16×16, 实际应上采样到目标尺寸
        out = F.interpolate(out, size=(224, 224), mode='bilinear',
                           align_corners=self.align_corners)

        return out


class CDHead(nn.Module):
    """
    变化检测头 — 轻量封装，支持多种解码器。
    """

    def __init__(self,
                 in_channels=[768, 768, 768, 768],
                 channels=512,
                 num_classes=2,
                 decoder_type='upernet'):
        super().__init__()

        if decoder_type == 'upernet':
            self.decoder = UPerHead(
                in_channels=in_channels,
                channels=channels,
                num_classes=num_classes
            )
        elif decoder_type == 'linear':
            # 简单线性分类头 (对比基线)
            self.decoder = nn.Sequential(
                nn.Conv2d(in_channels[-1], channels, kernel_size=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, num_classes, kernel_size=1)
            )
        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(self, feats):
        return self.decoder(feats)
