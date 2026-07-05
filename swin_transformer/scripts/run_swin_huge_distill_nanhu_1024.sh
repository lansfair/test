#!/usr/bin/env bash
# set -euo pipefail
cat ~/.bashrc
echo "初始化conda环境："
# export PATH="/root/miniconda3/bin:$PATH" >> ~/.bashrc
source ~/.bashrc
conda init
source /root/miniconda3/bin/activate
conda activate olmoearth
echo "初始化结束"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "TQ_GPU_NUM: $TQ_GPU_NUM"
echo "MASTER_ADDR: $MASTER_ADDR"
# printenv 
sleep 15s

# modified by zhoujiwen: 使用GPU 6和7进行7B蒸馏训练。
# cd "$(dirname "$0")/.."
export PYTHONPATH="${PYTHONPATH:-.}:."
# export CUDA_VISIBLE_DEVICES=0,1,2,3  # modified by zhoujiwen: 指定空闲GPU，避免占用4和5。

echo "[ssl-feature-distill] workspace: $(pwd)"
# echo "[ssl-feature-distill] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
test -f /mnt/ht2-nas2/00-model/00-zhoujw/dinov3-swin-distill-7B-with-gram/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth || { echo "[ssl-feature-distill][ERROR] missing teacher checkpoint"; exit 1; }  # modified by zhoujiwen
# test -d /mnt/htzzb2/EO_Pretrian_Data_global/Million-AID/extracted/train/train/ || { echo "[ssl-feature-distill][ERROR] missing train dataset dir"; exit 1; }  # modified by zhoujiwen
# test -d /mnt/htzzb2/EO_Pretrian_Data_global/Million-AID/extracted/test/test/ || { echo "[ssl-feature-distill][ERROR] missing val dataset dir"; exit 1; }  # modified by zhoujiwen

# python - <<'PY'
# import torch
# if not torch.cuda.is_available():
#     raise SystemExit("CUDA is not available. Refusing to run on CPU.")
# print(f"CUDA devices: {torch.cuda.device_count()} | current: {torch.cuda.get_device_name(0)}")
# PY


torchrun --nnodes=$WORLD_SIZE \
  --node_rank=$RANK \
  --master_addr=$MASTER_ADDR \
  --nproc_per_node=$TQ_GPU_NUM \
  --master_port $MASTER_PORT dinov3/train/train.py \
  --config-file dinov3/configs/train/swin_huge_feature_distill_nanhu.yaml \
  --output-dir /mnt/si000523ygkv/00-model/dinov3-distill-outputs/swin_base_vitl16_ssl_feature_distill_GE+IN22k+ZJSlice1024_8nodes "$@"
  # --output-dir outputs/swin_huge_distill_rs+imagenet_6nodes "$@"
