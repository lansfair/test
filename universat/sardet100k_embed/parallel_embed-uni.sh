#! /usr/bin/bash

function get_parent() {
    local n=$1
    local dir=$(pwd)
    while [ "$n" -gt 0 ]; do
        dir=$(dirname "$dir")
        n=$((n - 1))
    done
    echo "$dir"
}

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

# 自动读取归档目录绝对路径，非必要不修改。  
# export MM_ARCHIVE_HOME="$(get_parent 6)"

# 自动读取归档数据目录路径，非必要不修改。
# export MM_ARCHIVE_DATA_HOME="$MM_ARCHIVE_HOME/dat"
export MM_ARCHIVE_DATA_HOME="/mnt/ht2-nas2/EO_test/openmmlab-archive/dat"

# 自动读取归档预训练骨干网络权重目录路径，非必要不修改。
# export MM_ARCHIVE_CKPT_HOME="$MM_ARCHIVE_HOME/pretrained"
export MM_ARCHIVE_CKPT_HOME="/mnt/htzzb2/EO_test/00-zhumx/Working/checkpoints/universat-chpts/pretrained_chpts"

# export MM_ARCHIVE_EMBED_HOME="/mnt/htzzb2/EO_test/fh/embed"
export MM_ARCHIVE_EMBED_HOME="/mnt/htzzb2/EO_test/00-zhumx/Working/outputs/openmmlab_archive_work_dirs/embed"
# WORK_DIR="/mnt/htzzb2/EO_test/00-zhumx/Working/outputs/openmmlab_archive_work_dirs/$(basename "$MMSEG_ROOT")/$(basename "$UNIVERSAT_PROJECT")/$(basename "$PASTIS_PROJECT")/universat_pastisr_embeddings"
# mkdir -p "$WORK_DIR"

# Miniconda3 安装路径，非必要不修改。
export CONDA_HOME="/mnt/ht2-nas2/EO_test/miniconda3"

# Miniconda3 虚拟环境名称，确保此环境可以正常运行你的项目。
# export CONDA_ENV_NAME="${1:-"openmm-base"}"
export CONDA_ENV_NAME="${1:-"zz-openmm"}"

# 指定使用的 GPU 数量，整数。
export NUM_GPUS="${2:-2}"

# 启动配置名称。
# export CONFIG_NAME="${3:-"universat-base_1xb8-50e_dior-rgb-embed-linear"}"
export CONFIG_NAME="${3:-"extract_embeddings_dior_universat-base"}"

MMSEG_ROOT="/mnt/htzzb2/EO_test/00-zhumx/Working/Codes/openmmlab/openmmlab_archive/mmseg"

SRCDIR=$PWD
cd "$(get_parent 3)"

export PYTHONPATH="$MMSEG_ROOT:${PYTHONPATH:-}"
# export PYTHONPATH=".":"$PYTHONPATH"
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"

EMBED_OUTPUT_DIR="${MM_ARCHIVE_EMBED_HOME}/$(basename "${SRCDIR}")/${CONFIG_NAME}"
mkdir -p "$EMBED_OUTPUT_DIR"

GPU_LIST=$(seq -s, 0 $((NUM_GPUS - 1)))

echo "=========================================="
echo "MMSEG_ROOT: $MMSEG_ROOT"
echo "Config: $UNIVERSAT_PROJECT/configs/${CONFIG_NAME}.py"
# echo "GPUs: $GPUS"
# echo "NNODES: $NNODES"
# echo "NODE_RANK: $NODE_RANK"
# echo "MASTER_ADDR: $MASTER_ADDR"
# echo "MASTER_PORT: $MASTER_PORT"
echo "WorkDir: $EMBED_OUTPUT_DIR"
echo "PYTHONPATH: $PYTHONPATH"
echo "Extra args: $@"
echo "=========================================="

echo "========================================"
echo "  conda env  : $CONDA_ENV_NAME"
echo "  num gpus   : $NUM_GPUS ($GPU_LIST)"
echo "  config     : $CONFIG_NAME"
echo "========================================"

CUDA_VISIBLE_DEVICES="$GPU_LIST" \
    $PYINTERPRETER -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    "${SRCDIR}/tools/extract_embeddings.py" \
    "${SRCDIR}/configs/${CONFIG_NAME}.py" \
    --output-root "$EMBED_OUTPUT_DIR"
