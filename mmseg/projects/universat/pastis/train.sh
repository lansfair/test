#! /usr/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")" && pwd > /dev/null

# 自动读取归档目录绝对路径，非必要不修改。  
# export MM_ARCHIVE_HOME=$(echo "$PWD" | cut -d '/' -f 1-5)

# 自动读取归档数据目录路径，非必要不修改。
# export MM_ARCHIVE_DATA_HOME="$MM_ARCHIVE_HOME/dat"
export MM_ARCHIVE_DATA_HOME="/mnt/ht2-nas2/EO_test/openmmlab-archive/dat"

# 自动读取归档预训练骨干网络权重目录路径，非必要不修改。
# export MM_ARCHIVE_CKPT_HOME="$MM_ARCHIVE_HOME/src/v1/mmseg/pretrained"
export MM_ARCHIVE_CKPT_HOME="/mnt/htzzb2/EO_test/00-zhumx/Working/checkpoints/universat-chpts/pretrained_chpts"

# Miniconda3 安装路径，非必要不修改。
# export CONDA_HOME="$HOME/miniconda3"
export CONDA_HOME="/mnt/ht2-nas2/EO_test/miniconda3"

# Miniconda3 虚拟环境名称，确保此环境可以正常运行你的项目。
export CONDA_ENV_NAME='zz-openmm'

# 指定训练使用的 GPU 索引，格式：0, 1, 2..., N。
export CUDA_VISIBLE_DEVICES='0, 1'
                              
# 训练启动配置名称。
export CONFIG_NAME='universat-base_pastis_lp'


# SRCDIR: /mnt/htzzb2/EO_test/00-zhumx/Working/Codes/openmmlab/openmmlab_archive/mmseg/projects/universat/pastis
# SRCDIR=$PWD
# cd "$PWD/../../../"

MMSEG_ROOT="/mnt/htzzb2/EO_test/00-zhumx/Working/Codes/openmmlab/openmmlab_archive/mmseg"
cd "$MMSEG_ROOT"


# 需要同时把基础 UniverSat 项目和 PASTIS 项目加入 PYTHONPATH
# export PYTHONPATH=".":"$PWD/projects/universat":"$PWD/projects/universat/pastis":"$PYTHONPATH"
# export PYTHONPATH="$PWD":"$PWD/projects/universat":"$PWD/projects/universat/universat":"$PWD/projects/universat/pastis":"$PYTHONPATH"
export PYTHONPATH="$MMSEG_ROOT":"$PYTHONPATH"

PYINTERPRETER="$CONDA_HOME/envs/$CONDA_ENV_NAME/bin/python3"

# WORK_DIR="$MM_ARCHIVE_HOME/work_dirs/"$(basename $SRCDIR)"/$CONFIG_NAME"
WORK_DIR="/mnt/htzzb2/EO_test/00-zhumx/Working/outputs/openmmlab_archive_work_dirs/"$(basename $MMSEG_ROOT)"/$CONFIG_NAME"
mkdir -p "$WORK_DIR"

# $PYINTERPRETER -c "import projects.universat.universat.models.data_preprocessors.UniverSatDataPreprocessor; import universat_pastis; print('imports ok')"
# $PYINTERPRETER "$PWD/tools/train.py" "$SRCDIR/configs/$CONFIG_NAME.py" --work-dir "$WORK_DIR"
$PYINTERPRETER "$MMSEG_ROOT/tools/train.py" \
    "$MMSEG_ROOT/projects/universat/pastis/configs/$CONFIG_NAME.py" \
    --work-dir "$WORK_DIR"
