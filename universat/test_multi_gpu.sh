#! /usr/bin/bash
#
# Multi-GPU distributed testing script for UniverSat projects.
# Uses torchrun to launch testing on multiple GPUs.
#
# Usage:
#   bash test_multi_gpu.sh                    # Use default NPROC
#   bash test_multi_gpu.sh 4                  # Use 4 GPUs
#   NPROC=2 CHECKPOINT=epoch_50.pth bash test_multi_gpu.sh
#

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

# ---------------------------------------------------------------------------
# Environment paths (same as single-GPU test.sh)
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
NPROC=${NPROC:-$(nvidia-smi -L | wc -l)}

# 指定测试使用的 GPU 索引，格式：0,1,2...,N。
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

# ---------------------------------------------------------------------------
# Test config
# ---------------------------------------------------------------------------

# 测试启动配置名称，权重文件要求相同命名。
export CONFIG_NAME='pastisr_universat-base_seg'

# 权重文件路径。默认查找 work_dirs 下最新 checkpoint，可手动覆盖。
CHECKPOINT=${CHECKPOINT:-"$MM_ARCHIVE_HOME/work_dirs/$(basename $PWD)/$CONFIG_NAME/best_mIoU_iter_*.pth"}
# 如果通配符匹配多个，取第一个
CHECKPOINT=$(ls -t $CHECKPOINT 2>/dev/null | head -n 1)

if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
    echo "ERROR: Checkpoint not found: $CHECKPOINT"
    echo "Please set CHECKPOINT env var or ensure training has produced a .pth file."
    exit 1
fi

# ---------------------------------------------------------------------------
# Launch testing
# ---------------------------------------------------------------------------

SRCDIR=$PWD
cd "$PWD/../../../"

export PYTHONPATH=".":"$PYTHONPATH"
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"

WORK_DIR="$MM_ARCHIVE_HOME/work_dirs/"$(basename $SRCDIR)"/${CONFIG_NAME}_multi_test"
mkdir -p "$WORK_DIR"

echo "========================================"
echo "Multi-GPU Testing Launch"
echo "  Config:     $CONFIG_NAME"
echo "  Checkpoint: $CHECKPOINT"
echo "  GPUs:       $NPROC"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  Work dir:   $WORK_DIR"
echo "========================================"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    "$PWD/tools/test.py" \
    "$SRCDIR/configs/$CONFIG_NAME.py" \
    "$CHECKPOINT" \
    --work-dir "$WORK_DIR" \
    --launcher pytorch \
    "$@"
