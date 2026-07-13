import os
import argparse
from PIL import Image
from tqdm import tqdm

def stat_tif_info(folder_path):
    # 筛选所有 tif/tiff 格式文件
    tif_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.tif', '.tiff', '.png'))
    ]
    if not tif_files:
        print("文件夹下未找到 TIFF 文件")
        return

    global_min = float('inf')
    global_max = -float('inf')
    results = []

    for filename in tqdm(tif_files, desc="处理进度"):
        file_path = os.path.join(folder_path, filename)
        try:
            with Image.open(file_path) as img:
                w, h = img.size
                extrema = img.getextrema()

                # 兼容单通道/多通道图像，取像素极值
                if isinstance(extrema[0], tuple):
                    img_min = min(chan[0] for chan in extrema)
                    img_max = max(chan[1] for chan in extrema)
                else:
                    img_min, img_max = extrema

                results.append((filename, w, h, img_min, img_max))
                # 更新全局像素极值
                global_min = min(global_min, img_min)
                global_max = max(global_max, img_max)

        except Exception as e:
            tqdm.write(f"跳过损坏文件 {filename}: {str(e)}")

    # 输出统计结果
    print("\n===== 统计结果 =====")
    print(f"有效文件数: {len(results)}")
    print(f"全局像素最小值: {global_min}")
    print(f"全局像素最大值: {global_max}")
    print("\n各文件详情:")
    for name, w, h, imin, imax in results:
        print(f"{name}: 尺寸 {w}x{h}, 像素范围 [{imin}, {imax}]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="统计 TIFF 图像尺寸与像素极值")
    parser.add_argument("--folder", required=True, help="TIFF 图像文件夹路径")
    args = parser.parse_args()
    stat_tif_info(args.folder)