import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

def main(folder_path):
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
    if not tif_files:
        print("未找到 TIFF 文件")
        return

    pixel_count = defaultdict(int)
    total_pixels = 0

    for filename in tqdm(tif_files, desc="统计像素占比"):
        try:
            with Image.open(os.path.join(folder_path, filename)) as img:
                arr = np.array(img)
                values, counts = np.unique(arr, return_counts=True)
                for v, c in zip(values, counts):
                    pixel_count[v] += c
                total_pixels += arr.size
        except Exception as e:
            tqdm.write(f"跳过损坏文件 {filename}: {e}")

    # 按像素值从小到大输出占比
    print("\n像素值  占比")
    for val in sorted(pixel_count.keys()):
        print(f"{val:>6}  {pixel_count[val]/total_pixels:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    main(args.folder)