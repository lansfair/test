# -*- coding: utf-8 -*-
"""PCA降维到RGB + 可视化 (适配3000样本)."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA


# ========== 第一部分：PCA降维并保存为RGB jsonl ==========

def pca_to_rgb_jsonl(input_path, output_path, n_components=3):
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    records = []
    embeddings = []
    
    print(f"Reading: {input_path}")
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            embeddings.append(record["embedding"])
    
    X = np.array(embeddings, dtype=np.float32)
    print(f"Loaded {len(records)} samples, original shape: {X.shape}")
    
    # 检查原始维度
    if X.shape[1] != 768:
        print(f"Warning: expected 768 dims, got {X.shape[1]}")
    
    # PCA降维
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    # 归一化到0-255
    X_min = X_pca.min(axis=0)
    X_max = X_pca.max(axis=0)
    X_rgb = (X_pca - X_min) / (X_max - X_min + 1e-8)
    X_rgb = (X_rgb * 255).astype(np.uint8)
    
    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for i, record in enumerate(records):
            new_record = {k: v for k, v in record.items() if k != "embedding"}
            new_record["embedding"] = X_rgb[i].tolist()
            f.write(json.dumps(new_record, ensure_ascii=False) + "\n")
    
    print(f"PCA done. New shape: {X_rgb.shape}")
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"Total: {pca.explained_variance_ratio_.sum():.4f}")
    print(f"Saved RGB jsonl to: {output_path}")
    return X_rgb


# ========== 第二部分：可视化 ==========

def load_rgb_jsonl(path):
    records = []
    embeddings = []
    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            emb = record["embedding"]
            # 确保是列表
            if isinstance(emb, str):
                emb = json.loads(emb)
            embeddings.append(emb)
            ids.append(record.get("id", "unknown"))
    
    data = np.array(embeddings, dtype=np.float32)
    print(f"Loaded RGB data shape: {data.shape}")
    
    # 形状检查：必须是 (N, 3)
    if len(data.shape) == 1:
        # 如果是一维，可能是 (N*3,) 或 (N,)
        if data.shape[0] % 3 == 0:
            data = data.reshape(-1, 3)
            print(f"Reshaped to: {data.shape}")
        else:
            raise ValueError(f"Cannot reshape data of shape {data.shape} to (N, 3)")
    elif len(data.shape) == 2 and data.shape[1] != 3:
        raise ValueError(f"Expected (N, 3), got {data.shape}")
    
    return records, data, ids


def find_best_grid(n):
    sqrt_n = int(np.sqrt(n))
    best_h, best_w = 1, n
    best_diff = n - 1
    for h in range(sqrt_n, 0, -1):
        if n % h == 0:
            w = n // h
            diff = abs(w - h)
            if diff < best_diff:
                best_diff = diff
                best_h, best_w = h, w
    return best_h, best_w


def visualize_rgb_grid(rgb_data, save_path="rgb_grid.png"):
    N = rgb_data.shape[0]
    H, W = find_best_grid(N)
    print(f"Grid shape: {H}x{W}={N}")
    
    # 确保是 0-1 范围
    if rgb_data.max() > 1.0:
        img = rgb_data / 255.0
    else:
        img = rgb_data
    
    img = img.reshape(H, W, 3)
    
    fig, ax = plt.subplots(figsize=(W/10, H/10))
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"RGB Grid ({H}x{W}={N})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"RGB grid saved: {save_path}")


def visualize_scatter2d(rgb_data, labels=None, save_path="scatter2d.png"):
    fig, ax = plt.subplots(figsize=(10, 8))
    rgb_norm = rgb_data / 255.0 if rgb_data.max() > 1.0 else rgb_data
    
    if labels is not None:
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for idx, lab in enumerate(unique_labels):
            mask = np.array(labels) == lab
            ax.scatter(rgb_data[mask, 0], rgb_data[mask, 1],
                      c=[colors[idx]], label=str(lab), alpha=0.6, s=30)
        ax.legend()
    else:
        ax.scatter(rgb_data[:, 0], rgb_data[:, 1],
                  c=rgb_norm, alpha=0.7, s=40)
    
    ax.set_xlabel("R (PC1)")
    ax.set_ylabel("G (PC2)")
    ax.set_title("RGB Scatter 2D")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"2D scatter saved: {save_path}")


def visualize_scatter3d(rgb_data, labels=None, save_path="scatter3d.png"):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")
    rgb_norm = rgb_data / 255.0 if rgb_data.max() > 1.0 else rgb_data
    
    if labels is not None:
        unique_labels = sorted(set(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        for idx, lab in enumerate(unique_labels):
            mask = np.array(labels) == lab
            ax.scatter(rgb_data[mask, 0], rgb_data[mask, 1], rgb_data[mask, 2],
                      c=[colors[idx]], label=str(lab), alpha=0.6, s=30)
        ax.legend()
    else:
        ax.scatter(rgb_data[:, 0], rgb_data[:, 1], rgb_data[:, 2],
                  c=rgb_norm, alpha=0.7, s=40)
    
    ax.set_xlabel("R (PC1)")
    ax.set_ylabel("G (PC2)")
    ax.set_zlabel("B (PC3)")
    ax.set_title("RGB Scatter 3D")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"3D scatter saved: {save_path}")


def visualize_rgb_histogram(rgb_data, save_path="rgb_histogram.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = ["R (PC1)", "G (PC2)", "B (PC3)"]
    colors = ["#e74c3c", "#2ecc71", "#3498db"]
    
    for i in range(3):
        axes[i].hist(rgb_data[:, i], bins=50, color=colors[i],
                    edgecolor="black", alpha=0.7)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Frequency")
        axes[i].axvline(rgb_data[:, i].mean(), color="black",
                       linestyle="--", label=f"mean={rgb_data[:,i].mean():.1f}")
        axes[i].legend()
    
    plt.suptitle("RGB Channel Distributions", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Histogram saved: {save_path}")


def visualize_pairplot(rgb_data, labels=None, save_path="pairplot.png"):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    names = ["R", "G", "B"]
    rgb_norm = rgb_data / 255.0 if rgb_data.max() > 1.0 else rgb_data
    
    for i in range(3):
        for j in range(3):
            ax = axes[i, j]
            if i == j:
                ax.hist(rgb_data[:, i], bins=40, color="steelblue",
                       edgecolor="black", alpha=0.7)
                ax.set_title(names[i])
            else:
                if labels is not None:
                    unique_labels = sorted(set(labels))
                    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                    for idx, lab in enumerate(unique_labels):
                        mask = np.array(labels) == lab
                        ax.scatter(rgb_data[mask, j], rgb_data[mask, i],
                                  c=[colors[idx]], label=str(lab), alpha=0.5, s=15)
                else:
                    ax.scatter(rgb_data[:, j], rgb_data[:, i],
                              c=rgb_norm, alpha=0.5, s=15)
                ax.set_xlabel(names[j])
                ax.set_ylabel(names[i])
            ax.grid(True, linestyle="--", alpha=0.3)
    
    plt.suptitle("RGB Pair Plot", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Pairplot saved: {save_path}")


# ========== 主入口 ==========

def main(input_768_path, output_rgb_path, vis_output_dir, label_key=None):
    # Step 1: PCA降维 + 保存RGB jsonl
    print("=" * 50)
    print("Step 1: PCA to RGB")
    print("=" * 50)
    pca_to_rgb_jsonl(input_768_path, output_rgb_path)
    
    # Step 2: 可视化
    print("\n" + "=" * 50)
    print("Step 2: Visualization")
    print("=" * 50)
    
    vis_output_dir = Path(vis_output_dir)
    vis_output_dir.mkdir(parents=True, exist_ok=True)
    
    records, rgb_data, ids = load_rgb_jsonl(output_rgb_path)
    
    labels = None
    if label_key:
        labels = [r.get(label_key, "unknown") for r in records]
    
    # 1. RGB网格 (3000 -> 50x60)
    visualize_rgb_grid(rgb_data, save_path=vis_output_dir / "rgb_grid.png")
    
    # 2. 2D散点
    visualize_scatter2d(rgb_data, labels=labels,
                       save_path=vis_output_dir / "scatter2d.png")
    
    # 3. 3D散点
    visualize_scatter3d(rgb_data, labels=labels,
                       save_path=vis_output_dir / "scatter3d.png")
    
    # 4. 直方图
    visualize_rgb_histogram(rgb_data,
                           save_path=vis_output_dir / "histogram.png")
    
    # 5. Pairplot
    visualize_pairplot(rgb_data, labels=labels,
                      save_path=vis_output_dir / "pairplot.png")
    
    print(f"\nAll visualizations saved to: {vis_output_dir}")


if __name__ == "__main__":
    main(
        input_768_path="/root/olmoearth/copernicusfm1.jsonl",
        output_rgb_path="/root/olmoearth/embeddings_pca_rgb_new1.jsonl",
        vis_output_dir="/root/olmoearth/rgb_visualizations",
        label_key=None
    )