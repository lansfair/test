"""
把 dinov3_backbone.py 的 DINOv3BackboneMmseg (ViT-L/16 + DINOv3_Adapter，多尺度)
接到 mmrotate 1.x (mmengine) 上。

- backbone: 包一层 DINOv3Adapter0，注册进 mmrotate.registry.MODELS。
  返回 4 个尺度特征 (stride 4/8/16/32，各 1024 通道) 的 tuple，给 FPN 用。

在 mmrotate 1.x config 里用 custom_imports=dict(imports=['dinov3_mmrotate0']) 引入。
"""
import os
import sys

import torch
import torch.nn as nn
from mmcv.transforms import BaseTransform

# --- dinov3 源码上 path ---
# 优先环境变量 DINOV3_SRC (run_train.sh 会导出), 否则回退到本机默认路径
DINOV3_SRC = os.environ.get(
    'DINOV3_SRC', '/mnt/ht2-nas2/00-model/00-ds/dinov3-swin')
if DINOV3_SRC not in sys.path:
    sys.path.insert(0, DINOV3_SRC)

from dinov3_backbone import DINOv3BackboneMmseg  # noqa: E402

from mmrotate.registry import MODELS, TRANSFORMS  # noqa: E402

# LVD1689M ViT-L/16 官方 hub 工厂超参 (见 dinov3/hub/backbones.py:dinov3_vitl16)
_VITL_LVD1689M = dict(
    pos_embed_rope_base=100.0,
    pos_embed_rope_min_period=None,
    pos_embed_rope_max_period=None,
    pos_embed_rope_normalize_coords='separate',
    pos_embed_rope_rescale_coords=2,
    pos_embed_rope_dtype='fp32',
    norm_layer='layernormbf16',
    n_storage_tokens=4,
    mask_k_bias=True,
)

_DEFAULT_CKPT = '/mnt/ht2-nas2/EO_test/xyz/Dinov3_ORCNN/dinov3-swin/dinov3/weights/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'


@TRANSFORMS.register_module()
class RegularizeRotatedBoxes(BaseTransform):
    """将 pipeline 中的旋转框统一到指定角度约定。

    ``ConvertBoxType(qbox -> rbox)`` 最终依赖 ``cv2.minAreaRect``，不同
    OpenCV 版本可能返回 [0, pi/2] 等非规范角度。该变换必须放在所有几何
    增强之后、PackDetInputs 之前，使 GT 与 bbox coder 使用同一个约定。

    Args:
        pattern: RotatedBoxes.regularize_boxes 支持的 ``oc``、``le90``
            或 ``le135``。本项目使用 ``le90`` (mmrotate le90: 角度
            [-90, 90) 且宽为长边)。
        box_keys: results 中需要规范化的旋转框字段。
        min_size: 宽高合法性的下限；非法框直接报错，避免污染优化器。
    """

    def __init__(self,
                 pattern='le90',
                 box_keys=('gt_bboxes',),
                 min_size=1e-4):
        if pattern not in ('oc', 'le90', 'le135'):
            raise ValueError(f'Unsupported angle pattern: {pattern}')
        self.pattern = pattern
        self.box_keys = tuple(box_keys)
        self.min_size = float(min_size)

    def transform(self, results):
        for key in self.box_keys:
            boxes = results.get(key)
            if boxes is None or len(boxes) == 0:
                continue
            if not hasattr(boxes, 'regularize_boxes'):
                raise TypeError(
                    f"results['{key}'] must be RotatedBoxes before "
                    'RegularizeRotatedBoxes. Insert ConvertBoxType first, '
                    f'but got {type(boxes).__name__}.')

            # regularize_boxes 会原地更新 RotatedBoxes.tensor，同时处理
            # 角度周期、长短边互换和 le90 区间。
            boxes.regularize_boxes(pattern=self.pattern)
            tensor = boxes.tensor

            finite = torch.isfinite(tensor).all()
            valid_size = (tensor[..., 2:4] > self.min_size).all()
            valid_convention = tensor.new_tensor(True, dtype=torch.bool)
            if self.pattern == 'le90':
                eps = 1e-6
                angles = tensor[..., 4]
                valid_angle = ((angles >= -torch.pi / 2 - eps) &
                               (angles < torch.pi / 2)).all()
                width_is_long_edge = (
                    tensor[..., 2] + eps >= tensor[..., 3]).all()
                valid_convention = valid_angle & width_is_long_edge

            if not bool(finite & valid_size & valid_convention):
                img_path = results.get('img_path', '<unknown>')
                bad = (~torch.isfinite(tensor).all(dim=-1)) | (
                    tensor[..., 2:4] <= self.min_size).any(dim=-1)
                if self.pattern == 'le90':
                    bad = bad | (tensor[..., 4] < -torch.pi / 2 - 1e-6) | (
                        tensor[..., 4] >= torch.pi / 2) | (
                        tensor[..., 2] + 1e-6 < tensor[..., 3])
                raise ValueError(
                    f'Invalid rotated GT after {self.pattern} '
                    f'regularization in {img_path}: {tensor[bad]}')

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'pattern={self.pattern!r}, box_keys={self.box_keys!r}, '
                f'min_size={self.min_size})')


@MODELS.register_module()
class DINOv3Adapter0(nn.Module):
    """mmrotate 1.x 适配版 DINOv3 ViT-L/16 + Adapter 多尺度 backbone。

    forward 返回 (f4, f8, f16, f32)，每个 1024 通道，喂给 FPN
    (in_channels=[1024,1024,1024,1024])。
    """

    def __init__(self,
                 arch='vit_large',
                 patch_size=16,
                 checkpoint=_DEFAULT_CKPT,
                 freeze_backbone=True,
                 finetune_vit=False,
                 vit_cfg_overrides=None):
        super().__init__()
        overrides = dict(_VITL_LVD1689M)
        if vit_cfg_overrides:
            overrides.update(vit_cfg_overrides)
        self.net = DINOv3BackboneMmseg(
            arch=arch,
            patch_size=patch_size,
            checkpoint=checkpoint,
            freeze_backbone=freeze_backbone,
            finetune_vit=finetune_vit,
            vit_cfg_overrides=overrides,
        )

    def init_weights(self):
        pass  # weights loaded inside DINOv3BackboneMmseg (build_model_for_eval)

    def forward(self, x):
        return self.net(x)  # tuple(f4, f8, f16, f32)
