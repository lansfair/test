import os
import rasterio
import numpy as np
from tqdm import tqdm

# ========== 仅需修改路径 ==========
input_dir = r"/mnt/htzzb2/EO_test/wj1/Ningbo_2m_work_dirs/infer_results/dinov3_large_adapter_upernet_full_backbone0.1_50e"
output_dir = r"/mnt/htzzb2/EO_test/wj1/Ningbo_2m_work_dirs/infer_results/dinov3_large_adapter_upernet_full_backbone0.1_50e_vis"
# ===================================

# 原始标签的color map
colormap = {
    1:  (255, 0, 0),    # 红
    2:  (0, 255, 0),    # 绿
    3:  (0, 0, 255),    # 蓝
    4:  (255, 255, 0),  # 黄
    5:  (255, 0, 255),  # 品红
    6:  (0, 255, 255),  # 青
    7:  (255, 128, 0),  # 橙
    8:  (128, 0, 255),  # 紫
    9:  (0, 128, 0),    # 深绿
    10: (139, 69, 19),  # 棕
    17: (255, 192, 203),# 浅粉
    28: (128, 128, 128) # 灰
}

# 推理结果的可视化color map
colormap = {
    0:  (255, 0, 0),    # 红
    1:  (0, 255, 0),    # 绿
    2:  (0, 0, 255),    # 蓝
    3:  (255, 255, 0),  # 黄
    4:  (255, 0, 255),  # 品红
    5:  (0, 255, 255),  # 青
    6:  (255, 128, 0),  # 橙
    7:  (128, 0, 255),  # 紫
    8:  (0, 128, 0),    # 深绿
    9: (139, 69, 19),  # 棕
    10: (255, 192, 203),# 浅粉
    11: (128, 128, 128) # 灰
}

os.makedirs(output_dir, exist_ok=True)
file_list = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tif', '.tiff', '.png'))]

for fname in tqdm(file_list, desc="彩色映射中"):
    with rasterio.open(os.path.join(input_dir, fname)) as src:
        arr = src.read(1)
        profile = src.profile.copy()  # 完整保留原始坐标、分辨率、坐标系等元数据

    # 初始化三通道全黑背景
    rgb = np.zeros((3, *arr.shape), dtype=np.uint8)
    # 应用色彩映射
    for val, (r, g, b) in colormap.items():
        mask = arr == val
        rgb[0][mask] = r
        rgb[1][mask] = g
        rgb[2][mask] = b

    # 更新输出配置：三通道、8位、PNG格式
    profile.update(count=3, dtype='uint8', driver='PNG')
    out_name = os.path.splitext(fname)[0] + '.png'
    
    with rasterio.open(os.path.join(output_dir, out_name), 'w', **profile) as dst:
        dst.write(rgb)

print("处理完成")
