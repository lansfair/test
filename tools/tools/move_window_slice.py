import os
import argparse
from PIL import Image
from tqdm import tqdm

def slide_crop_dataset(data_root, output_root, window_size=128):
    step = window_size  # 步长等于窗口大小，无重叠切片
    splits = ["train", "val"]

    for split in splits:
        img_dir = os.path.join(data_root, split, "images")
        mask_dir = os.path.join(data_root, split, "masks")
        out_split_dir = f"{split}_{window_size}"
        out_img_dir = os.path.join(output_root, out_split_dir, "images")
        out_mask_dir = os.path.join(output_root, out_split_dir, "masks")

        os.makedirs(out_img_dir, exist_ok=True)
        os.makedirs(out_mask_dir, exist_ok=True)

        tif_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".tif", ".tiff"))]
        if not tif_files:
            print(f"{split} 集无 TIFF 文件，跳过")
            continue

        total_crops = 0
        for filename in tqdm(tif_files, desc=f"处理 {split} 集"):
            img_path = os.path.join(img_dir, filename)
            mask_path = os.path.join(mask_dir, filename)

            if not os.path.exists(mask_path):
                tqdm.write(f"警告：掩码文件缺失 {split}/{filename}")
                continue

            try:
                with Image.open(img_path) as img, Image.open(mask_path) as mask:
                    w, h = img.size
                    # 仅保留完整的 128x128 窗口，边缘不足部分丢弃
                    xs = range(0, w - window_size + 1, step)
                    ys = range(0, h - window_size + 1, step)

                    name_stem, ext = os.path.splitext(filename)
                    for x in xs:
                        for y in ys:
                            box = (x, y, x + window_size, y + window_size)
                            img_crop = img.crop(box)
                            mask_crop = mask.crop(box)

                            new_name = f"{name_stem}_{x}_{y}{ext}"
                            img_crop.save(os.path.join(out_img_dir, new_name))
                            mask_crop.save(os.path.join(out_mask_dir, new_name))
                            total_crops += 1

            except Exception as e:
                tqdm.write(f"跳过损坏文件 {split}/{filename}: {str(e)}")

        print(f"{split} 集完成：共生成 {total_crops} 张 128x128 切片\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TIFF图像与掩码无重叠滑窗切片")
    parser.add_argument("--data-root", required=True, help="原始数据集根目录")
    parser.add_argument("--output-root", required=True, help="输出数据集根目录")
    args = parser.parse_args()

    slide_crop_dataset(args.data_root, args.output_root)