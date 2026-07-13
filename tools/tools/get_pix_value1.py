import os
import argparse
from PIL import Image
from tqdm import tqdm

def main(folder_path):
    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
    if not tif_files:
        return

    global_min = float('inf')
    global_max = -float('inf')

    for filename in tqdm(tif_files):
        try:
            with Image.open(os.path.join(folder_path, filename)) as img:
                ex = img.getextrema()
                # 兼容单通道与多通道
                if isinstance(ex[0], tuple):
                    img_min = min(c[0] for c in ex)
                    img_max = max(c[1] for c in ex)
                else:
                    img_min, img_max = ex

                global_min = min(global_min, img_min)
                global_max = max(global_max, img_max)
        except Exception:
            pass

    print(global_min, global_max)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    args = parser.parse_args()
    main(args.folder)