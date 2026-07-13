import os
import shutil
import rasterio
import numpy as np
from tqdm import tqdm

# ========== 仅需修改这两个路径 ==========
data_root = r"/mnt/ht2-nas2/EO_test/wj1/datasets/Ningbo-2m/val"
new_data_root = r"/mnt/ht2-nas2/EO_test/wj1/datasets/Ningbo-2m/val_smaller_than_0.5"
# ========================================

# 拼接子目录路径
img_dir = os.path.join(data_root, "images")
mask_dir = os.path.join(data_root, "masks")
new_img_dir = os.path.join(new_data_root, "images")
new_mask_dir = os.path.join(new_data_root, "masks")

# 自动创建目标目录
os.makedirs(new_img_dir, exist_ok=True)
os.makedirs(new_mask_dir, exist_ok=True)

# 获取images下所有tif文件
tif_list = [f for f in os.listdir(img_dir) if f.lower().endswith((".tif", ".tiff"))]

moved_count = 0
for fname in tqdm(tif_list, desc="筛选处理中"):
    img_path = os.path.join(img_dir, fname)
    # 读取三通道影像
    with rasterio.open(img_path) as src:
        arr = src.read()  # 维度: (3, 高度, 宽度)
    
    # 计算三通道全为255的像素占比
    full_white_mask = (arr == 255).all(axis=0)
    white_ratio = full_white_mask.sum() / full_white_mask.size
    
    # 占比超过50%则同步移动image和mask
    if white_ratio > 0.5:
        shutil.copy2(img_path, os.path.join(new_img_dir, fname))
        # 移动同名mask，不存在则跳过
        mask_path = os.path.join(mask_dir, fname)
        if os.path.exists(mask_path):
            shutil.copy2(mask_path, os.path.join(new_mask_dir, fname))
        moved_count += 1

print(f"处理完成，共移动 {moved_count} 组影像")
