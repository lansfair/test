#!/usr/bin/env bash
set -euo pipefail

# modified by zhoujiwen: 使用GPU 6和7进行7B蒸馏训练。
cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:."
export CUDA_VISIBLE_DEVICES=0,1,2,3  # modified by zhoujiwen: 指定空闲GPU，避免占用4和5。

echo "[ssl-feature-distill] workspace: $(pwd)"
echo "[ssl-feature-distill] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
test -f /mnt/ht2_nas2/00-model/00-zhoujw/dinov3-swin-distill-7B-with-gram/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth || { echo "[ssl-feature-distill][ERROR] missing teacher checkpoint"; exit 1; }  # modified by zhoujiwen
test -d /mnt/qh2-nas3/EO_Pretrian_Data_global/Million-AID/extracted/train/train/ || { echo "[ssl-feature-distill][ERROR] missing train dataset dir"; exit 1; }  # modified by zhoujiwen
test -d /mnt/qh2-nas3/EO_Pretrian_Data_global/Million-AID/extracted/test/test/ || { echo "[ssl-feature-distill][ERROR] missing val dataset dir"; exit 1; }  # modified by zhoujiwen

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Refusing to run on CPU.")
print(f"CUDA devices: {torch.cuda.device_count()} | current: {torch.cuda.get_device_name(0)}")
PY



torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" dinov3/train/train.py \
  --config-file dinov3/configs/train/swin_base_feature_distill_vitl16_flatjpgtxt.yaml \
  --output-dir outputs/swin_base_vitl16_ssl_feature_distill_imagenet "$@"
