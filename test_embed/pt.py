# -*- coding: utf-8 -*-
import sys
import torch
import numpy as np
from pathlib import Path

def check_pt_structure(pt_path):
    pt_path = Path(pt_path)
    if not pt_path.exists():
        print(f"错误：文件不存在 -> {pt_path}")
        return False

    print(f"===== 检查文件: {pt_path} =====")
    data = torch.load(pt_path, map_location="cpu")

    # 1. 顶层数据类型
    print(f"\n1. 顶层数据类型: {type(data)}")
    if not isinstance(data, dict):
        print("❌ 不满足可视化要求：pt顶层不是字典，没有 embeddings / labels 键")
        return False

    # 2. 打印所有key
    all_keys = list(data.keys())
    print(f"\n2. 文件内所有键: {all_keys}")
    required_keys = {"embeddings", "labels"}
    missing = required_keys - set(all_keys)
    if missing:
        print(f"❌ 缺失必要键: {missing}，无法送入可视化脚本")
        return False
    print("✅ 包含必需的 embeddings、labels 键")

    # 3. 查看embeddings信息
    emb = data["embeddings"]
    print(f"\n3. embeddings 信息")
    print(f"   tensor shape: {emb.shape}")
    print(f"   dtype: {emb.dtype}")
    print(f"   device: {emb.device}")
    if len(emb.shape) != 2:
        print(f"⚠️ 警告：embeddings不是二维[N, C]，当前维度 {emb.shape}，可视化要求 [样本数,768]")
    else:
        print(f"   样本数 N = {emb.shape[0]}, 特征维度 = {emb.shape[1]}")

    # 4. 查看labels信息
    lab = data["labels"]
    print(f"\n4. labels 信息")
    print(f"   tensor shape: {lab.shape}")
    print(f"   dtype: {lab.dtype}")
    unique_lab = torch.unique(lab)
    print(f"   所有类别标签: {unique_lab.tolist()}")
    print(f"   类别总数: {len(unique_lab)}")
    if len(lab.shape) != 1:
        print(f"⚠️ 警告：labels不是一维向量，当前维度 {lab.shape}，要求 [N]")

    # 5. 样本数量匹配校验
    n_emb = emb.shape[0]
    n_lab = lab.shape[0]
    if n_emb == n_lab:
        print(f"\n✅ 样本数量匹配：emb={n_emb}, label={n_lab}")
    else:
        print(f"\n❌ 样本数量不匹配！emb数量{n_emb} != label数量{n_lab}")
        return False

    print("\n===== 校验完成：该pt文件可以直接输入可视化脚本 =====")
    return True

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        pt_file = sys.argv[1]
    else:
        # 替换成你要检查的pt默认路径
        pt_file = "/mnt/ht2-nas2/users_project/Common/olmoearth_embeddings/dino_v3_dinov3_vitl16_data/dino_v3_dinov3_vitl16/m_so2sat/train.pt"
    check_pt_structure(pt_file)