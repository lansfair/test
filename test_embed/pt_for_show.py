# -*- coding: utf-8 -*-
import torch
import numpy as np
import sys

def extract_single_img_16patch(src_pt_path, dst_pt_path, img_idx=0):
    data = torch.load(src_pt_path, map_location="cpu")
    emb_4d = data["embeddings"]   # [B,16,16,768]
    label_3d = data["labels"]    # [B,64,64]
    B = emb_4d.shape[0]
    assert 0 <= img_idx < B, f"图片索引{img_idx}超出范围，总图片数{B}"

    # 取单张图
    img_emb = emb_4d[img_idx]    # [16,16,768]
    img_label_map = label_3d[img_idx]  # [64,64]

    # 1. Patch展平：16*16=256个Patch [256,768]
    flat_emb = img_emb.reshape(-1, 768)

    # 2. 每个Patch对应4×4像素，取区域主类别
    patch_pixel_size = 4
    flat_labels = []
    for i in range(16):
        for j in range(16):
            # Patch对应标签图像素范围
            y_start = i * patch_pixel_size
            y_end = (i + 1) * patch_pixel_size
            x_start = j * patch_pixel_size
            x_end = (j + 1) * patch_pixel_size
            patch_label_region = img_label_map[y_start:y_end, x_start:x_end]

            # 过滤ignore label -1
            valid_pixels = patch_label_region[patch_label_region != -1]
            if len(valid_pixels) == 0:
                flat_labels.append(-1)
                continue
            # 取出现最多的类别
            values, counts = torch.unique(valid_pixels, return_counts=True)
            main_cls = int(values[torch.argmax(counts)])
            flat_labels.append(main_cls)

    flat_labels = torch.tensor(flat_labels, dtype=torch.int64)

    # 保存适配可视化代码的pt
    save_dict = {
        "embeddings": flat_emb,
        "labels": flat_labels
    }
    torch.save(save_dict, dst_pt_path)

    print(f"提取完成，第{img_idx}张图16×16 Patch已展平")
    print(f"emb shape: {flat_emb.shape}")
    print(f"label shape: {flat_labels.shape}")
    print(f"包含类别: {torch.unique(flat_labels).tolist()}")

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        src = sys.argv[1]
        dst = sys.argv[2]
        idx = int(sys.argv[3]) if len(sys.argv) >= 4 else 0
    else:
        src = "/mnt/xxx/your_seg.pt"
        dst = "/mnt/xxx/single_16patch_flat.pt"
        idx = 0
    extract_single_img_16patch(src, dst, idx)