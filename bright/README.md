# olmoearth_cd — 服务器多卡版

OlmoEarth v1.1-Base 光学+SAR 双端变化检测，BRIGHT 4 类建筑损伤评估，支持单卡 / 多卡 DDP 自适应。

---

## 环境依赖

```
torch torchvision tqdm albumentations opencv-python pillow numpy imagecodecs tifffile
```

---

## 快速开始

### 1. 放置权重

在 `weights/` 下放两个文件：
- `config.json` — OlmoEarth 模型配置
- `weights.pth` — OlmoEarth v1.1-Base 预训练权重 (982 MB)

### 2. 修改训练参数

编辑 `train.py` 顶部的 `CONFIG` 字典：

```python
CONFIG = {
    'data_root':   '/data/datasets/BRIGHT',   # ← 改成你的数据集路径
    'batch_size':  4,
    'epochs':      100,
    'finetune':    True,
    ...
}
```

### 3. 启动训练

```bash
# 单卡
python train.py

# 4 卡
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train.py

# 8 卡
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 train.py
```

### 4. 断点续跑

```bash
python train.py --resume ./outputs/2026-06-18_10-00-00
```

---

## 模型架构

```
前时相光学 → OlmoEarth ViT (12ch S2_L2A, 768-dim)
后时相 SAR  → OlmoEarth ViT (2ch S1_GRD, 768-dim)
     ↓                    ↓
   diff → FeaturePyramid → UperNet → 4类损伤
```

全参微调：差分学习率 (ViT=5e-6, Embed=5e-5, Decoder=1e-4)

---

## 本地最佳成绩 (BRIGHT Test, v3 通道Padding)

| Test F1 | IoU | P | R |
|:--|:--|:--|:--|
| 0.538 | 0.400 | 0.470 | 0.726 |

---

## 自包含说明

本项目内嵌 `olmoearth_pretrain/` 源码，复制到任何服务器直接可跑，零外部依赖。
