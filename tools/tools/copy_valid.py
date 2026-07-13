import os
import argparse
import shutil
from PIL import Image
from tqdm import tqdm

def filter_copy_dataset(data_root, output_root):
    splits = ["train", "val"]

    for split in splits:
        img_dir = os.path.join(data_root, split, "images")
        mask_dir = os.path.join(data_root, split, "masks")
        out_img_dir = os.path.join(output_root, split, "images")
        out_mask_dir = os.path.join(output_root, split, "masks")

        # 创建输出目录
        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        # 筛选所有 tif/tiff 文件
        tif_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".tif", ".tiff"))]
        if not tif_files:
            print(f"{split} 集 images 下无 TIFF 文件，跳过")
            continue

        skip_count = 0
        for filename in tqdm(tif_files, desc=f"处理 {split} 集"):
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            try:
                # 判断图像是否全白（所有像素所有通道均为255）
                with Image.open(img_path) as img:
                    extrema = img.getextrema()
                    # 兼容单通道与多通道图像
                    ch_extrema = extrema if isinstance(extrema[0], tuple) else [extrema]
                    is_all_white = all(ch_min == 255 for ch_min, _ in ch_extrema)
                    if is_all_white:
                        skip_count += 1
                        continue

                # 非全白则复制原图与对应掩码
                shutil.copy(img_path, os.path.join(out_img_dir, filename))
                if os.path.exists(mask_path):
                    shutil.copy(mask_path, os.path.join(out_mask_dir, filename))
                else:
                    tqdm.write(f"警告：掩码文件不存在 {split}/{filename}")

            except Exception as e:
                tqdm.write(f"跳过损坏文件 {split}/{filename}: {str(e)}")
                skip_count += 1

        print(f"{split} 集完成：保留 {len(tif_files)-skip_count} 张 | 排除 {skip_count} 张\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="筛选非全白TIFF并复制数据集")
    parser.add_argument("--data-root", required=True, help="原始数据集根目录")
    parser.add_argument("--output-root", required=True, help="输出数据集根目录")
    args = parser.parse_args()

    filter_copy_dataset(args.data_root, args.output_root)