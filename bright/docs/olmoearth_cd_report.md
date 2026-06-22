# olmoearth_cd — OlmoEarth 光学+SAR 变化检测

> 状态：⚠️ 收益递减，已归档 | 数据集：DFC2025 (BRIGHT) | 最佳 Test F1：**0.538**

---

## 📋 项目定位

探索 **OlmoEarth 单一预训练框架**在光学+SAR 变化检测上的上限。与 optical_sar_cd（DINOv2 + OlmoEarth）不同，本项目**两端都使用 OlmoEarth 预训练权重**，旨在验证 OlmoEarth 的遥感预训练能否独立支撑变化检测任务。

> 结论：OlmoEarth 单独不够，但作为 SAR 分支补充 DINOv2 光学效果良好。

---

## 🏗️ 架构

```
前时相光学 RGB → OlmoEarth S2_L2A ViT (冻结) → 多尺度特征 [B,768,*,*]
后时相 SAR 1ch → OlmoEarth S1_GRD ViT (冻结) → 多尺度特征 [B,768,*,*]
                                                    ↓ abs diff
                                                   UPerNet
                                                    ↓
                                            4 类损伤图 [B,4,H,W]
```

| 组件 | 详情 |
|:--|:--|
| 光学编码器 | OlmoEarth v1.1-Base S2_L2A (12ch), 冻结 |
| SAR 编码器 | OlmoEarth v1.1-Base S1_GRD (2ch), 冻结 |
| 融合方式 | 逐层 abs diff |
| Decoder | UPerNet |
| 参数量 | ~86M (双 ViT 冻结, 仅训练 decoder) |

---

## 📊 版本历程

| 版本 | SAR 策略 | 光学策略 | F1 | 备注 |
|:--|:--|:--|:--|:--|
| v1 | 切片 1ch→proj | 切片 3ch→proj | 0.462 | 基线，切片破坏语义 |
| v2 | 切片 + 预训练 proj weight | 同 | 0.464 | 微涨 |
| v2 全参 | 切片 + 全参微调 | 同 | 0.484 | 全参有提升但有限 |
| **v3** | **通道 Padding (1→2ch 补零)** | **12ch 完整输入** | **0.538** 🏆 | 保留预训练语义 |

### v3 最终成绩

| 指标 | Test |
|:--|:--|
| **F1** | **0.538** |
| **IoU** | **0.400** |
| **Precision** | 0.470 |
| **Recall** | 0.726 |

> 特征：Recall 高 (0.73) 但 Precision 低 (0.47)。模型对损伤敏感（不太漏），但误报多。这是冻结 backbone + 小数据集的典型困境——decoder 学会了"找变化"，但没有足够数据教会它"忽视非变化"。

---

## 🧠 关键经验

### 1. 通道 Padding > 通道切片
v1/v2 用切片（只取前 3ch）破坏了 S2_L2A 12 通道色带的预训练耦合。v3 完整 12ch 输入 + 缺失波段填零，F1 从 0.484 → 0.538 (+0.054)，方向确认正确。

### 2. OlmoEarth 单独作为变化检测 backbone 有上限
双 OlmoEarth 冻结训练 → F1 上限约 0.54，远低于 optical_sar_cd v2（DINOv2 光学 + OlmoEarth SAR）的 0.65。DINOv2 的通用视觉预训练在变化检测判别上比 OlmoEarth 遥感预训练更强。

**原因推测：** OlmoEarth 的预训练任务是多模态对齐（光学↔SAR 配对），不是"变化判别"。DINOv2 的自监督特征学习更通用，在下游"找差异"时表现更好。

### 3. 冻结 backbone 是最大瓶颈
v2 全参微调从 0.464 → 0.484 (+0.02)，提升有限。OlmoEarth ViT-B (12 层) 在几千张 BRIGHT 图上全参微调的收益不如 DINOv2 (24 层)。

### 4. v3 之后收益递减
+0.054 有意义，但接近框架上限。继续调参最多再挤 0.02~0.03。果断转向 optical_sar_cd 的 DINOv2+OlmoEarth 混合方案是正确的。

---

## 📁 文件结构

```
olmoearth_cd/
├── train.py                  # 训练入口
├── models/
│   ├── olmoearth_cd.py       # 双 OlmoEarth 模型
│   └── cd_head.py            # UPerNet decoder
├── weights/
│   └── weights.pth           # OlmoEarth v1.1-Base (982MB)
├── outputs/
│   ├── v1_frozen/            # F1=0.462
│   ├── v2_finetune/          # F1=0.484
│   └── v3_padding/           # F1=0.538 🏆
├── docs/
│   └── olmoearth_cd_report.md  # 本文档
├── dinov3/                    # DINOv3 源码
└── olmoearth_nn/              # OlmoEarth nn 模块
```

---

## 🎯 参考价值

- ✅ 通道 Padding 策略验证
- ✅ OlmoEarth 单独作为 CD backbone 的**上限参考**（~0.54）
- ✅ 作为 optical_sar_cd 的**对照实验**，量化了 DINOv2 光学 vs OlmoEarth 光学的差距（0.65 vs 0.54）

## ❌ 不建议继续投资

框架内收益已递减。OlmoEarth 的价值在于为 DINOv2 提供 SAR 分支，而非独立支撑变化检测。
