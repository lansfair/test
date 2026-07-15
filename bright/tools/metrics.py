"""
变化检测指标计算器 — 支持多分类（mIoU）和二分类（F1）。
"""

import numpy as np
import torch
from typing import Dict


class CDMetrics:
    """
    多分类语义分割指标（mIoU / F1 / Precision / Recall）。
    对二分类也完全兼容（num_classes=2）。
    """

    def __init__(self, num_classes: int = 2, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth
        self.reset()

    def reset(self):
        """重置混淆矩阵。"""
        self.confusion = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, pred: torch.Tensor, target: torch.Tensor):
        """
        累加混淆矩阵。

        Args:
            pred: [B, H, W] 预测类别
            target: [B, H, W] 真实标签（含 ignore_index）
        """
        pred = pred.cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()

        # 只考虑有效像素（忽略 255）
        valid = target < self.num_classes
        pred = pred[valid]
        target = target[valid]

        # 累加混淆矩阵
        for t, p in zip(target, pred):
            self.confusion[t, p] += 1

    def compute(self) -> Dict[str, float]:
        """计算多分类指标。"""
        cm = self.confusion
        n = self.num_classes

        # 每类的 TP, FP, FN
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        # 每类的 IoU
        iou_per_class = tp / (tp + fp + fn + self.smooth)
        miou = float(iou_per_class.mean())

        # 每类的 F1
        precision = tp / (tp + fp + self.smooth)
        recall = tp / (tp + fn + self.smooth)
        f1_per_class = 2 * precision * recall / (precision + recall + self.smooth)
        mf1 = float(f1_per_class.mean())

        # 总体准确率
        accuracy = float(np.diag(cm).sum() / (cm.sum() + self.smooth))

        # 二分类兼容输出（变化类 = 第 1 类）
        if n >= 2:
            bin_iou = float(iou_per_class[1]) if n > 1 else miou
            bin_f1 = float(f1_per_class[1]) if n > 1 else mf1
            bin_precision = float(precision[1]) if n > 1 else float(precision.mean())
            bin_recall = float(recall[1]) if n > 1 else float(recall.mean())
        else:
            bin_iou = miou
            bin_f1 = mf1
            bin_precision = float(precision.mean())
            bin_recall = float(recall.mean())

        return {
            'miou': miou,
            'f1': mf1,
            'iou': bin_iou,
            'precision': bin_precision,
            'recall': bin_recall,
            'accuracy': accuracy,
            'iou_per_class': iou_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
        }


def compute_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """单次计算指标（非累计）。"""
    metrics = CDMetrics(num_classes=max(pred.max(), target.max()) + 1)
    metrics.update(torch.from_numpy(pred), torch.from_numpy(target))
    return metrics.compute()
