#!/usr/bin/env bash
# Example launch script for extracting UniverSat embeddings on DIOR.
#
# Run from the MMSegmentation root (the directory that contains ``tools/``).
# The UniverSat project must be visible on PYTHONPATH.

set -euo pipefail

# 禁用 TorchInductor 编译优化，回退到 eager 模式
export TORCHDYNAMO_DISABLE=1

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

# ---------------------------------------------------------------------------
# User-configurable paths
# ---------------------------------------------------------------------------
export MM_ARCHIVE_DATA_HOME="/mnt/ht2-nas2/EO_test/openmmlab-archive/dat"
export MM_ARCHIVE_CKPT_HOME="/mnt/htzzb2/EO_test/00-zhumx/Working/checkpoints/universat-chpts/pretrained_chpts"

export CONDA_HOME="/mnt/ht2-nas2/EO_test/miniconda3"
export CONDA_ENV_NAME='zz-openmm'

# ---------------------------------------------------------------------------
# Distributed settings
# ---------------------------------------------------------------------------
# 使用不同的变量名避免与$@冲突
CONFIG_NAME="${1:-extract_embeddings_dior_universat-base}"
GPUS="${2:-2}"
# 移除第3个参数的位置占用，让$@只包含额外参数
shift 2 2>/dev/null || true

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_PORT="${PORT:-29501}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# ---------------------------------------------------------------------------
# Resolve MMSeg root and project paths
# ---------------------------------------------------------------------------
MMSEG_ROOT="/mnt/htzzb2/EO_test/00-zhumx/Working/Codes/openmmlab/openmmlab_archive/mmseg"
UNIVERSAT_PROJECT="$MMSEG_ROOT/projects/universat"

# export PYTHONPATH="$MMSEG_ROOT:$UNIVERSAT_PROJECT:$UNIVERSAT_PROJECT/pastis:${PYTHONPATH:-}"
export PYTHONPATH="$MMSEG_ROOT:${PYTHONPATH:-}"

# 使用conda环境中的Python
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"
TORCHRUN="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/torchrun"

# 如果conda环境中没有torchrun，使用系统torchrun
if [ ! -f "$TORCHRUN" ]; then
    TORCHRUN="$(which torchrun 2>/dev/null || echo 'torchrun')"
fi

WORK_DIR="/mnt/htzzb2/EO_test/00-zhumx/Working/outputs/openmmlab_archive_work_dirs/$(basename "$MMSEG_ROOT")/$(basename "$UNIVERSAT_PROJECT")/universat_dior_embeddings"
mkdir -p "$WORK_DIR"

echo "=========================================="
echo "MMSEG_ROOT: $MMSEG_ROOT"
echo "UNIVERSAT_PROJECT: $UNIVERSAT_PROJECT"
echo "Config: $UNIVERSAT_PROJECT/configs/${CONFIG_NAME}.py"
echo "GPUs: $GPUS"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WorkDir: $WORK_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "Extra args: $@"
echo "=========================================="

# 检查配置文件是否存在
CONFIG_FILE="${UNIVERSAT_PROJECT}/configs/${CONFIG_NAME}.py"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# 执行多卡训练
$TORCHRUN \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$MASTER_PORT \
    "${UNIVERSAT_PROJECT}/tools/extract_embeddings.py" \
    "$CONFIG_FILE" \
    --output-root "$WORK_DIR" \
    --splits train val test \
    --batch-size 1 \
    --tile-size 0 \
    --device auto \
    --precision bf16 \
    --skip-existing \
    "$@"