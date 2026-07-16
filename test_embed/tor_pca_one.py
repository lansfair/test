# -*- coding: utf-8 -*-
"""PCA to RGB + Visualization for .pt files (embeddings + labels)."""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA


def load_pt_data(pt_path):
    """Load .pt file containing dict with 'embeddings' and 'labels'."""
    data = torch.load(pt_path, map_location="cpu")

    embeddings = data["embeddings"]          # [N, 768], bfloat16
    labels = data["labels"]                  # [N], int64

    # Convert bfloat16 to float32 for PCA (sklearn does not support bfloat16)
    if embeddings.dtype == torch.bfloat16:
        embeddings = embeddings.to(torch.float32)

    embeddings = embeddings.numpy()            # [N, 768] float32
    labels = labels.numpy()                    # [N] int64

    print(f"Loaded from .pt: embeddings shape={embeddings.shape}, dtype={embeddings.dtype}")
    print(f"Labels shape={labels.shape}, unique labels={np.unique(labels)}")

    return embeddings, labels


def pca_to_rgb(embeddings, n_components=3):
    """PCA reduce [N, 768] to [N, 3] and scale to 0-255."""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(embeddings)      # [N, 3]

    X_min = X_pca.min(axis=0)
    X_max = X_pca.max(axis=0)
    X_rgb = (X_pca - X_min) / (X_max - X_min + 1e-8)
    X_rgb = (X_rgb * 255).astype(np.uint8)

    print(f"PCA done. Shape: {X_rgb.shape}")
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"Total: {pca.explained_variance_ratio_.sum():.4f}")

    return X_rgb


def visualize_rgb_grid(rgb_data, save_path="rgb_grid.png", title="RGB Grid", fixed_size=None):
    N = rgb_data.shape[0]

    if fixed_size is not None:
        H, W = fixed_size
        target = H * W
        if N > target:
            rgb_data = rgb_data[:target]
            N = target
        pad_count = target - N
        if pad_count > 0:
            pad = np.zeros((pad_count, 3), dtype=rgb_data.dtype)
            rgb_data = np.concatenate([rgb_data, pad], axis=0)
    else:
        H = int(np.sqrt(N))
        while N % H != 0 and H > 1:
            H -= 1
        W = N // H
        target = H * W
        if N > target:
            rgb_data = rgb_data[:target]
        elif N < target:
            pad = np.zeros((target - N, 3), dtype=rgb_data.dtype)
            rgb_data = np.concatenate([rgb_data, pad], axis=0)

    if rgb_data.max() > 1.0:
        img = rgb_data / 255.0
    else:
        img = rgb_data

    img = img.reshape(H, W, 3)

    fig, ax = plt.subplots(figsize=(max(W/10, 4), max(H/10, 4)))
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"{title} ({H}x{W}, N={N})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"RGB grid saved: {save_path}")


def visualize_rgb_by_label(rgb_data, labels, output_dir, max_per_label=256, grid_hw=(16, 16)):
    """Group by label and draw separate RGB grid for each label."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for i, lab in enumerate(labels):
        groups[int(lab)].append(i)

    print(f"Found {len(groups)} labels: {sorted(groups.keys())}")

    for lab, indices in sorted(groups.items()):
        indices = indices[:max_per_label]
        group_rgb = rgb_data[indices]
        n = len(indices)

        safe_label = str(lab).replace("/", "_").replace("\\", "_")
        save_path = output_dir / f"rgb_grid_label_{safe_label}.png"

        visualize_rgb_grid(
            group_rgb,
            save_path=save_path,
            title=f"Label: {lab} (n={n})",
            fixed_size=grid_hw
        )


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


def main(pt_path, vis_output_dir):
    print("=" * 50)
    print("Step 1: Load .pt and PCA to RGB")
    print("=" * 50)

    embeddings, labels = load_pt_data(pt_path)
    rgb_data = pca_to_rgb(embeddings)

    print("" + "=" * 50)
    print("Step 2: Visualization")
    print("=" * 50)

    vis_output_dir = Path(vis_output_dir)
    vis_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall RGB grid
    visualize_rgb_grid(rgb_data, save_path=vis_output_dir / "rgb_grid_all.png",
                       title="All Samples", fixed_size=None)

    # 2. Label-grouped RGB grids (16x16, max 256 per label)
    print("" + "-" * 50)
    print("Generating label-grouped RGB grids (16x16, max 256 per label)...")
    print("-" * 50)
    visualize_rgb_by_label(rgb_data, labels,
                           vis_output_dir / "by_label",
                           max_per_label=256,
                           grid_hw=(16, 16))

    # 3. 2D scatter
    visualize_scatter2d(rgb_data, labels=labels,
                       save_path=vis_output_dir / "scatter2d.png")

    # 4. 3D scatter
    visualize_scatter3d(rgb_data, labels=labels,
                       save_path=vis_output_dir / "scatter3d.png")

    # 5. Histogram
    visualize_rgb_histogram(rgb_data,
                           save_path=vis_output_dir / "histogram.png")

    # 6. Pairplot
    visualize_pairplot(rgb_data, labels=labels,
                      save_path=vis_output_dir / "pairplot.png")

    print(f"All visualizations saved to: {vis_output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        pt_path = sys.argv[1]
    else:
        pt_path = "/mnt/ht2-nas2/users_project/Common/olmoearth_embeddings/dino_v3_dinov3_vitl16_data/dino_v3_dinov3_vitl16/m_so2sat/train.pt"

    vis_dir = "/mnt/qh2-nas3/EO_test/cyz/pastis-r_embed_olmoearth/"
    main(pt_path, vis_dir)