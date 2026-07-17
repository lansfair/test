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

# export MM_WORK_DIR_HOME="$(get_parent 7)"
export MM_WORK_DIR_HOME="/mnt/htzzb2/EO_test"

# 自动读取归档目录绝对路径，非必要不修改。  
export MM_ARCHIVE_HOME="$(get_parent 6)"

# 自动读取归档数据目录路径，非必要不修改。
export MM_ARCHIVE_DATA_HOME="$MM_ARCHIVE_HOME/dat"

# 自动读取归档预训练骨干网络权重目录路径，非必要不修改。
export MM_ARCHIVE_CKPT_HOME="$MM_ARCHIVE_HOME/pretrained"

# Miniconda3 安装路径，非必要不修改。
export CONDA_HOME="/mnt/ht2-nas2/EO_test/miniconda3"

# Miniconda3 虚拟环境名称，确保此环境可以正常运行你的项目。
export CONDA_ENV_NAME="${1:-"openmm-2m"}"

# 指定使用的 GPU 索引，格式：0, 1, 2..., N。
export CUDA_VISIBLE_DEVICES="${2:-0}"
                              
# 启动配置名称。
export CONFIG_NAME="${3:-"self-olmoearth-base-10m_1xb4-50e_m-sa-crop-type-s2-ft/self-olmoearth-base-10m_1xb4-50e_m-sa-crop-type-s2-ft_1e-4"}"


SRCDIR=$PWD
cd "$(get_parent 3)"
export PYTHONPATH=".":"$PYTHONPATH"
PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"
WORK_DIR="${MM_WORK_DIR_HOME}/openmmlab_work_dirs/mmseg/$(basename "${SRCDIR}")/${CONFIG_NAME}"
mkdir -p "$WORK_DIR"

$PYINTERPRETER "$PWD/tools/train.py" "$SRCDIR/configs/$CONFIG_NAME.py" --work-dir "$WORK_DIR"