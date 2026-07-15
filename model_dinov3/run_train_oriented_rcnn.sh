#!/bin/bash
# =============================================================================
# 启动脚本: DINOv3 ViT-L + Adapter + FPN + Oriented R-CNN on DIOR-R
#
# 用法:
#   bash run_train_orcnn.sh              # 单卡 (默认 GPU 0)
#   bash run_train_orcnn.sh 0,1          # 2 卡分布式
#   bash run_train_orcnn.sh 0,1,2,3      # 4 卡分布式
# =============================================================================
set -e

# === 本机路径 ===
THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMLAB_DIR="/mnt/ht2-nas2/EO_test/xyz/Dinov3_ORCNN/mmlab"
MMROTATE_DIR="${MMLAB_DIR}/mmrotate"
DINOV3_SWIN="/mnt/ht2-nas2/EO_test/xyz/Dinov3_ORCNN/dinov3-swin"
DINOV3_MMROTATE="/mnt/ht2-nas2/00-model/00-ds/dinov3_mmrotate"

CONFIG="${THIS_DIR}/dinov3_dior_oriented_rcnn_mmr0.py"
# WORK_DIR="${THIS_DIR}/work_dirs/dinov3_dior_oriented_rcnn"
WORK_DIR="/mnt/htzzb2/EO_test/xyz/work_dirs/dinov3_dior_oriented_rcnn"

# PYTHONPATH: 必须与 run_train.sh 完全一致, 否则 custom_imports / dinov3 会失败。
#   本目录       -> 供 custom_imports 导入 dinov3_mmrotate0 / dinov3_backbone
#   dinov3-swin  ->  dinov3 包源码 (from dinov3.models import ...)
#   dinov3_mmrotate / mmlab(mmcv, mmdetection, mmrotate) -> mmlab 系列
# 注意: 不要追加 ${PYTHONPATH}! 交互式 shell 里常含其它环境的 site-packages。
export PYTHONPATH="${THIS_DIR}:${DINOV3_SWIN}:${DINOV3_MMROTATE}:${MMLAB_DIR}/mmcv:${MMLAB_DIR}/mmdetection:${MMLAB_DIR}/mmrotate"
# dinov3_backbone.py 的 _resolve_dino_root() 会读取此环境变量
export DINOV3_SRC="${DINOV3_SWIN}"
 
GPUS="${1:-0}"
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
NUM_GPUS=${#GPU_ARRAY[@]}
export CUDA_VISIBLE_DEVICES="$GPUS"

echo "=========================================="
echo " Config:   $CONFIG"
echo " Work Dir: $WORK_DIR"
echo " GPUs:     $GPUS  (共 ${NUM_GPUS} 张)"
echo "=========================================="

cd "$MMROTATE_DIR"

if [ "$NUM_GPUS" -le 1 ]; then
    conda run -n mmrotate-dinov3 --no-capture-output python3 tools/train.py "$CONFIG" \
        --work-dir "$WORK_DIR"
else
    # 注意：不要直接用 `torchrun`！mmrotate 环境的 torchrun 脚本 shebang
    # 被错误地写成了 #!/.../envs/olmoearth/bin/python3.12，会强制用 olmoearth
    # 的解释器（其 setuptools 60.2.0 在 Python3.12 下损坏 -> distutils 缺失 ->
    # mmengine collect_env 崩溃）。改用 `python3 -m torch.distributed.run`，
    # 由 `conda run -n mmrotate python3` 保证解释器就是 mmrotate 的。
    conda run -n mmrotate-dinov3 --no-capture-output python3 -m torch.distributed.run \
        --nproc_per_node="$NUM_GPUS" \
        --master_port=29501 \
        tools/train.py \
        "$CONFIG" \
        --work-dir "$WORK_DIR" \
        --launcher pytorch
fi
