# UniverSat 在 PASTIS-R 上的评估

本项目评估 UniverSat 在 PASTIS-R 数据集上的语义分割性能。其结构遵循 `projects/copernicus/pastis` 和 `projects/olmoearth/pastis` 项目。

## 目录结构

```
pastis/
├── universat_pastis/           # PASTIS-R 组件的 Python 包
│   ├── datasets/pastis.py      # UniverSatPASTISDataset + 数据整理函数
│   ├── transforms/formatting.py
│   └── utils/norm.py
├── configs/
│   ├── universat-base_pastis_lp.py  # 线性探测（冻结骨干网络）
│   └── universat-base_pastis_ft.py  # 微调
├── train.sh
└── test.sh
```

骨干网络、解码头和数据预处理器复用自 `projects/universat/universat`。

## 数据布局

您的 PASTIS-R 目录应如下所示：

```
PASTIS-R/
  metadata.geojson              # ID_PATCH, Fold, dates-S2, dates-S1A
  DATA_S2/S2_{id}.npy           # T x 10 x H x W
  DATA_S1A/S1A_{id}.npy         # T x 3 x H x W
  ANNOTATIONS/TARGET_{id}.npy   # 1 x H x W 或 H x W
  NORM_S2_patch.json            # {"mean": [...], "std": [...]}
  NORM_S1_patch.json            # {"mean": [...], "std": [...]}
```

如果您没有归一化 JSON 文件，请先根据训练集拆分计算它们，或者在配置中临时设置 `norm_path=None`（不推荐用于实际评估）。

## 使用方法

在 `train.sh` / `test.sh` 中设置环境变量（特别是 `MM_ARCHIVE_DATA_HOME`、`MM_ARCHIVE_CKPT_HOME`、`CONDA_ENV_NAME` 和 `CUDA_VISIBLE_DEVICES`），然后运行：

```bash
cd projects/universat/pastis
bash train.sh
```

或者从 MMSegmentation 根目录手动运行：

```bash
export PYTHONPATH=".:$PWD/projects/universat:$PWD/projects/universat/pastis:$PYTHONPATH"
python tools/train.py \
    projects/universat/pastis/configs/universat-base_pastis_lp.py \
    --work-dir work_dirs/universat-base_pastis_lp
```

## 配置文件

- `universat-base_pastis_lp.py`：骨干网络冻结（`frozen_stages=0`），仅训练线性探测头。用于标准的 LP 评估。
- `universat-base_pastis_ft.py`：骨干网络解冻（`frozen_stages=-1`），使用小型卷积分割头对整个模型进行微调。

要切换到 UniverSat-Tiny，请将 `embed_dim` 改为 192，`num_heads` 改为 8，`block_type` 改为 `("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub")`（默认的 Base 配置已有 12 个 SA 块；Tiny 有 6 个）。

## 注意事项

- PASTIS-R 样本的时间序列长度可变。自定义的整理函数（`universat_pastis_collate`）会将每个模态及其日期张量填充到批次中的最大长度。
- 传递给骨干网络的输入字典包含两个模态张量（`s2`、`s1`）及其对应的日期张量（`s2_dates`、`s1_dates`）。
- `output_grid=128` 表示骨干网络输出 128 x 128 的令牌网格，与 PASTIS-R 原生的 128 x 128 分辨率匹配。
- `num_classes=20` 且 `ignore_index=19`：PASTIS-R 有 20 个标签（0=背景，1-18=作物类别，19=空洞）。空洞类别被忽略，背景被视为有效类别。如果您的标注约定不同，请相应调整。
