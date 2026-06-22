"""
预测可视化工具：olmoearth_cd 项目（BRIGHT 4 类建筑损伤）

用法:
    # 独立运行
    python tools/plot_predictions.py --weight outputs/best_model.pth --output outputs/figures/

    # 从 train.py 调用
    from tools.plot_predictions import visualize_predictions
    visualize_predictions(model, data_root, output_dir, device)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

DAMAGE_COLORS = {
    0: [0, 0, 0],        # no damage → black
    1: [0, 255, 0],      # minor → green
    2: [255, 255, 0],    # major → yellow
    3: [255, 0, 0],      # destroyed → red
    4: [128, 128, 128],  # ignore → gray
}
LABEL_NAMES = {0: 'No Damage', 1: 'Minor', 2: 'Major', 3: 'Destroyed'}


def colorize_label(label, num_classes=4):
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in DAMAGE_COLORS.items():
        if cls_id <= num_classes:
            rgb[label == cls_id] = color
    return rgb


def load_test_samples(data_root, num_samples=8, img_size=224):
    """加载 BRIGHT 测试集样本。"""
    from train import BrightCDDataset

    ds = BrightCDDataset(data_root, 'test', img_size, is_train=False)
    indices = np.linspace(0, len(ds) - 1, num_samples, dtype=int)
    return [ds[i] for i in indices]


def visualize_predictions(model, data_root, output_dir, device='cuda',
                          num_samples=8, img_size=224):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    print(f"[Predictions] Loading {num_samples} test samples...")
    samples = load_test_samples(data_root, num_samples, img_size)

    print("[Predictions] Running inference...")
    preds = []
    prob_damages = []  # 损伤概率热力图
    with torch.no_grad():
        for pre_opt, post_sar, _ in tqdm(samples, desc='Predict'):
            pre_opt = pre_opt.unsqueeze(0).to(device)
            post_sar = post_sar.unsqueeze(0).to(device)
            out = model(pre_opt, post_sar)
            probs = torch.softmax(out, dim=1)  # [1, 4, H, W]
            pred = out.argmax(dim=1)[0].cpu().numpy()
            prob_dmg = probs[0, 1:4].max(dim=0)[0].cpu().numpy()  # max(轻微,严重,摧毁)
            preds.append(pred)
            prob_damages.append(prob_dmg)

    print("[Predictions] Generating figure...")
    fig, axes = plt.subplots(num_samples, 5, figsize=(20, 4 * num_samples))

    for i, ((pre_opt, post_sar, label), pred, prob_dmg) in enumerate(zip(samples, preds, prob_damages)):
        # Optical
        opt = pre_opt.cpu().numpy().transpose(1, 2, 0)
        opt = opt * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        opt = np.clip(opt, 0, 1)

        # SAR
        sar = post_sar.cpu().numpy().squeeze(0)
        sar = np.clip(sar * 0.25 + 0.5, 0, 1)

        # GT
        gt = label.cpu().numpy()
        gt[label == 4] = 0
        gt_rgb = colorize_label(gt)

        # Pred
        pred_rgb = colorize_label(pred)

        axes[i, 0].imshow(opt)
        axes[i, 0].set_title('Pre-disaster (Optical)', fontsize=10)
        axes[i, 1].imshow(sar, cmap='gray')
        axes[i, 1].set_title('Post-disaster (SAR)', fontsize=10)
        axes[i, 2].imshow(pred_rgb)
        axes[i, 2].set_title('Prediction', fontsize=10)
        axes[i, 3].imshow(gt_rgb)
        axes[i, 3].set_title('Ground Truth', fontsize=10)
        im = axes[i, 4].imshow(prob_dmg, cmap='hot', vmin=0, vmax=1)
        axes[i, 4].set_title(f'Damage Prob (max={prob_dmg.max():.3f})', fontsize=10)
        for j in range(5):
            axes[i, j].axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'predictions.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Predictions] Saved: {save_path}")

    # Legend
    fig_l, ax_l = plt.subplots(figsize=(6, 1.5))
    patches = [plt.Rectangle((0,0),1,1,fc=np.array(DAMAGE_COLORS[i])/255) for i in range(4)]
    ax_l.legend(patches, [LABEL_NAMES[i] for i in range(4)], loc='center', ncol=4)
    ax_l.axis('off')
    legend_path = os.path.join(output_dir, 'legend.png')
    fig_l.savefig(legend_path, dpi=100, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', default='E:/change_detection/olmoearth_cd/outputs/best_model.pth')
    parser.add_argument('--data-root', default='E:/datasets/BRIGHT')
    parser.add_argument('--output', default='E:/change_detection/olmoearth_cd/outputs/figures')
    parser.add_argument('--num-samples', type=int, default=8)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    from models.olmoearth_cd import build_olmoearth_cd

    print(f"Loading model from {args.weight}...")
    model = build_olmoearth_cd(
        config_path='E:/change_detection/weights/config.json',
        weight_path='E:/change_detection/weights/weights.pth',
        num_classes=4, img_size=args.img_size,
    )
    model.load_state_dict(torch.load(args.weight, map_location='cpu', weights_only=True))
    model = model.to(args.device)

    visualize_predictions(model, args.data_root, args.output, args.device,
                          args.num_samples, args.img_size)
