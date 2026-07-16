import os
import csv
import random
from pathlib import Path

# 配置
root_dir = "/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/EuroSAT/tif"
train_ratio = 0.8
sample_max = 100  # 每个类别最多取前100张
seed = 42
random.seed(seed)

# 遍历所有类别文件夹
cls_folders = [d for d in Path(root_dir).iterdir() if d.is_dir()]

for cls_dir in cls_folders:
    cls_name = cls_dir.name
    all_tif_files = list(cls_dir.glob("*.tif"))
    if not all_tif_files:
        print(f"[{cls_name}] 无tif，跳过")
        continue
    
    # 只取前100张
    tif_files = all_tif_files[:sample_max]
    print(f"[{cls_name}] 实际采样数量：{len(tif_files)} (总文件数：{len(all_tif_files)})")

    # 打乱并切分
    random.shuffle(tif_files)
    split_point = int(len(tif_files) * train_ratio)
    train_tifs = tif_files[:split_point]
    val_tifs = tif_files[split_point:]

    # 只存文件名（不带路径）
    train_names = [[f.name] for f in train_tifs]
    val_names = [[f.name] for f in val_tifs]

    # 写入当前类别文件夹下的csv（无表头）
    train_csv = cls_dir / "train100.csv"
    val_csv = cls_dir / "val100.csv"

    with open(train_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(train_names)

    with open(val_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(val_names)

    print(f"[{cls_name}] train:{len(train_tifs)} val:{len(val_tifs)} 生成完成")