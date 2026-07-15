"""
绘制训练曲线 — 从 metrics.txt 提取指标，输出 Loss/F1/IoU/PR 曲线图。
自动从脚本所在位置推断 logs/ 和 outputs/ 路径。
"""

import os, sys, argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def parse_metrics(metrics_path):
    """
    解析 metrics.txt，格式（空格分隔）：
    epoch train_loss val_f1 val_iou val_precision val_recall
    """
    if not os.path.exists(metrics_path):
        print(f"  [ERROR] {metrics_path} not found")
        return None

    epoch_data = {}
    with open(metrics_path, 'r', encoding='ascii') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                ep = int(parts[0])
                epoch_data[ep] = {
                    'epoch': ep,
                    'train_loss': float(parts[1]),
                    'val_f1': float(parts[2]),
                    'val_iou': float(parts[3]),
                    'val_precision': float(parts[4]),
                    'val_recall': float(parts[5]),
                }
            except (ValueError, IndexError):
                continue

    if not epoch_data:
        print("  [ERROR] no valid data in metrics.txt")
        return None

    sorted_eps = sorted(epoch_data.keys())
    return {
        'epochs': sorted_eps,
        'train_loss': [epoch_data[e]['train_loss'] for e in sorted_eps],
        'val_f1': [epoch_data[e]['val_f1'] for e in sorted_eps],
        'val_iou': [epoch_data[e]['val_iou'] for e in sorted_eps],
        'val_precision': [epoch_data[e]['val_precision'] for e in sorted_eps],
        'val_recall': [epoch_data[e]['val_recall'] for e in sorted_eps],
    }


def plot_all(data, save_dir, prefix=''):
    os.makedirs(save_dir, exist_ok=True)
    epochs = np.array(data['epochs'])

    # ── Figure 1: Loss ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, data['train_loss'], 'b-', linewidth=1.5, label='Train Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{prefix}Training Loss' if prefix else 'Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}loss_curve.png'), dpi=150)
    plt.close(fig)

    # ── Figure 2: F1 + IoU ──
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, data['val_f1'], 'g-', linewidth=1.5, label='Val F1')
    ax1.plot(epochs, data['val_iou'], 'orange', linewidth=1.5, label='Val IoU')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Score')
    ax1.set_title(f'{prefix}Validation F1 & IoU' if prefix else 'Validation F1 & IoU')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    # 标注最佳 F1
    if data['val_f1']:
        best_idx = np.argmax(data['val_f1'])
        ax1.annotate(f'Best F1={data["val_f1"][best_idx]:.4f}',
                     xy=(epochs[best_idx], data['val_f1'][best_idx]),
                     xytext=(epochs[best_idx] + 2, data['val_f1'][best_idx] + 0.03),
                     fontsize=9, color='#2ca02c',
                     arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=1.2))
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}f1_iou_curve.png'), dpi=150)
    plt.close(fig)

    # ── Figure 3: Precision + Recall ──
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, data['val_precision'], 'c-', linewidth=1.5, label='Val Precision')
    ax.plot(epochs, data['val_recall'], 'm-', linewidth=1.5, label='Val Recall')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title(f'{prefix}Validation Precision & Recall' if prefix else 'Validation Precision & Recall')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'{prefix}precision_recall_curve.png'), dpi=150)
    plt.close(fig)

    print(f"  -> saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Plot training curves from metrics.txt')
    parser.add_argument('--metrics', default=None,
                        help='Path to metrics.txt (default: PROJECT_DIR/logs/metrics.txt)')
    parser.add_argument('--output-dir', default=None,
                        help='Output directory for figures (default: PROJECT_DIR/outputs/figures)')
    parser.add_argument('--prefix', default='', help='Filename prefix for figures')
    args = parser.parse_args()

    metrics_path = args.metrics or os.path.join(PROJECT_DIR, 'logs', 'metrics.txt')
    save_dir = args.output_dir or os.path.join(PROJECT_DIR, 'outputs', 'figures')

    plot_from_metrics(metrics_path, save_dir, args.prefix)


def plot_from_metrics(metrics_path, save_dir, prefix=''):
    """供 train.py 调用的接口。"""
    print(f"Loading: {metrics_path}")
    data = parse_metrics(metrics_path)
    if data is None:
        print(f"  [WARN] Could not parse {metrics_path}, skipping plots")
        return
    print(f"  epochs: {min(data['epochs'])} - {max(data['epochs'])}")
    plot_all(data, save_dir, prefix)


if __name__ == '__main__':
    main()
