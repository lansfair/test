"""Extraction config for UniverSat Base on DIOR.

This config is used by ``projects/universat/tools/extract_embeddings.py`` to
extract dense backbone embeddings for DIOR RGB images. DIOR is a detection
dataset, so its bounding boxes are converted to a semantic mask by
``UniverSatDIORDataset``; the mask is saved alongside the embedding but is not
used to train a head here.

Expected DIOR layout::

    ${MM_ARCHIVE_DATA_HOME}/DIOR/
      JPEGImages/
      Annotations/
      ImageSets/Main/{train,val,test}.txt

Run from the MMSegmentation root (the folder containing ``tools/``)::

    export PYTHONPATH=".:$PWD/projects/universat:$PYTHONPATH"
    export MM_ARCHIVE_DATA_HOME=/path/to/data
    export MM_ARCHIVE_CKPT_HOME=/path/to/checkpoints

    python projects/universat/tools/extract_embeddings.py \
        projects/universat/configs/extract_embeddings_dior_universat-base.py \
        --output-root work_dirs/universat_dior_embeddings \
        --splits train val test \
        --batch-size 1 \
        --tile-size 0

For distributed extraction launch it with ``torchrun``.
"""

# import os

# ---------------------------------------------------------------------------
# Imports and dataset
# ---------------------------------------------------------------------------
custom_imports = dict(
    # imports=['projects.universat.universat'],
    imports=["projects.universat.dior_embed.universat"],
    allow_failed_imports=False,
)

_base_ = ['./dataset/dior.py']

# ---------------------------------------------------------------------------
# Distributed training settings
# ---------------------------------------------------------------------------
# ``find_unused_parameters=True`` is required for DDP when parts of the model
# (e.g. frozen backbone layers) do not receive gradients in every iteration.
# ``model_wrapper_cfg=None`` uses mmengine's default DDP wrapper.
# 多卡
model_wrapper_cfg = None
find_unused_parameters = True


pretrained = '{{$MM_ARCHIVE_CKPT_HOME:checkpoints}}/model.safetensors'

# ---------------------------------------------------------------------------
# Model (only the backbone is used during extraction)
# ---------------------------------------------------------------------------
backbone_embed_dim = 768
modalities = ['rgb']

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(type='UniverSatDataPreprocessor'),
    backbone=dict(
        type='UniverSatBackbone',
        modalities=modalities,
        embed_dim=backbone_embed_dim,
        num_heads=12,
        patch_size=40,
        # DIOR images are ~800 x 800 px at ~0.5 m/px. With patch_size=40 m,
        # the natural latent side is 800 / (40 / 0.5) = 10 tokens.
        output_grid=10,
        block_type=('Bi_ACA_in', 'SAx12', 'Bilinear_out', 'CA_Sub'),
        n_registers=4,
        gating=True,
        frozen_stages=-1,
        # DIOR RGB is not in the UniverSat modality registry, so provide the
        # required metadata explicitly.
        wavelengths={'rgb': [0.665, 0.56, 0.49]},
        input_res={'rgb': 0.5},
        subpatches={'rgb': 1},
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained,
        ),
    ),
    decode_head=None,
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ---------------------------------------------------------------------------
# Runtime (minimal stubs required by MMSeg config loading)
# ---------------------------------------------------------------------------
default_scope = 'mmseg'
log_level = 'INFO'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
