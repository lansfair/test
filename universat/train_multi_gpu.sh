#! /usr/bin/bash
#
# Multi-GPU distributed training script for UniverSat projects.
# Uses torchrun (torch.distributed.launch) to launch training on multiple GPUs.
#
# Usage:
#   bash train_multi_gpu.sh          # Use default NPROC (all available GPUs)
#   bash train_multi_gpu.sh 4        # Use 4 GPUs
#   NPROC=2 bash train_multi_gpu.sh  # Use 2 GPUs via env var
#

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

# ---------------------------------------------------------------------------
# Environment paths (same as single-GPU train.sh)
# ---------------------------------------------------------------------------

# 自动读取归档目录绝对路径，非必要不修改。  
export MM_ARCHIVE_HOME=$(echo "$PWD" | cut -d '/' -f 1-5)

# 自动读取归档数据目录路径，非必要不修改。
export MM_ARCHIVE_DATA_HOME="$MM_ARCHIVE_HOME/dat"

# 自动读取归档预训练骨干网络权重目录路径，非必要不修改。
export MM_ARCHIVE_CKPT_HOME="$MM_ARCHIVE_HOME/src/v1/mmseg/pretrained"

# Miniconda3 安装路径，非必要不修改。
export CONDA_HOME="$HOME/miniconda3"

# Miniconda3 虚拟环境名称，确保此环境可以正常运行你的项目。
export CONDA_ENV_NAME='mmseg'

# ---------------------------------------------------------------------------
# Multi-GPU settings
# ---------------------------------------------------------------------------

# Number of GPUs / processes. Default: all GPUs visible to CUDA.
# Can be overridden by: NPROC=4 bash train_multi_gpu.sh
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}

# 指定训练使用的 GPU 索引，格式：0,1,2...,N。
# torchrun 会通过 LOCAL_RANK 自动分配，这里只需确保可见。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

# 训练启动配置名称。
export CONFIG_NAME='pastisr_universat-base_seg'

# Optional: auto-scale learning rate with batch size (linear scaling rule)
# Uncomment to enable: LR_SCALE enables MMSeg's auto_scale_lr feature
# export LR_SCALE_ENABLE=true
# export LR_SCALE_BASE_BATCH_SIZE=8

# ---------------------------------------------------------------------------
# Launch training
# ---------------------------------------------------------------------------

SRCDIR=$PWD
cd "$PWD/../../../"

export PYTHONPATH=".":"$PYTHONPATH"
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"

WORK_DIR="$MM_ARCHIVE_HOME/work_dirs/"$(basename $SRCDIR)"/${CONFIG_NAME}_multi"
mkdir -p "$WORK_DIR"

echo "========================================"
echo "Multi-GPU Training Launch"
echo "  Config:     $CONFIG_NAME"
echo "  GPUs:       $NPROC"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  Work dir:   $WORK_DIR"
echo "========================================"

# torchrun automatically sets RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    "$PWD/tools/train.py" \
    "$SRCDIR/configs/$CONFIG_NAME.py" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    "$@"
