"""Extraction config for UniverSat Base on Potsdam (RGB-only).

This config is used by ``projects/universat/tools/extract_embeddings.py`` to
extract dense backbone embeddings for downstream offline probes.

Expected Potsdam layout::

    ${MM_ARCHIVE_DATA_HOME}/potsdam/
      splits/train.json
      splits/val.json
      splits/test.json
      img_dir/      # RGB images
      ann_dir/      # annotation masks

Each JSON split file contains a list of dicts::

    [
      {
        "filenames": {"rgb": "img_dir/train/top_mosaic_09cm_area1.tif"},
        "ann": {"seg_map": "ann_dir/train/top_mosaic_09cm_area1.tif"},
        "height": 512,
        "width": 512
      },
      ...
    ]

Run from the MMSegmentation root::

    export PYTHONPATH=".:$PWD/projects/universat:$PYTHONPATH"
    python projects/universat/tools/extract_embeddings.py \
        projects/universat/configs/extract_embeddings_potsdam_universat-base.py \
        --output-root work_dirs/universat_potsdam_embeddings \
        --splits train val test
"""

import os

custom_imports = dict(
    imports=["projects.universat.universat"],
    allow_failed_imports=False,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
data_root = os.path.join(
    os.environ.get("MM_ARCHIVE_DATA_HOME", "data"), "potsdam"
)
pretrained = os.path.join(
    os.environ.get("MM_ARCHIVE_CKPT_HOME", "checkpoints"),
    "universat_base.safetensors",
)

crop_size = (512, 512)
num_classes = 5
ignore_index = 5  # void/background class
modalities = ["rgb"]

# Potsdam RGB images are uint8 [0, 255]; normalize to [0, 1].
norm_cfg = dict(
    mean={"rgb": [0.0, 0.0, 0.0]},
    std={"rgb": [255.0, 255.0, 255.0]},
)

train_pipeline = [
    dict(type="LoadMultimodalFromFile", modalities=modalities),
    dict(type="LoadAnnotations"),
    dict(type="NormalizeMultimodal", **norm_cfg),
    dict(type="PackUniverSatInputs"),
]

test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatSegDataset",
        modalities=modalities,
        data_root=data_root,
        split="splits/train.json",
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatSegDataset",
        modalities=modalities,
        data_root=data_root,
        split="splits/val.json",
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatSegDataset",
        modalities=modalities,
        data_root=data_root,
        split="splits/test.json",
        pipeline=test_pipeline,
    ),
)

# ---------------------------------------------------------------------------
# Model (only the backbone is used during extraction)
# ---------------------------------------------------------------------------
backbone_embed_dim = 768

model = dict(
    type="EncoderDecoder",
    data_preprocessor=dict(type="UniverSatDataPreprocessor"),
    backbone=dict(
        type="UniverSatBackbone",
        modalities=modalities,
        embed_dim=backbone_embed_dim,
        num_heads=12,
        patch_size=40,
        output_grid=36,  # 6 x 6 tokens for 512 x 512 Potsdam patches
        block_type=("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"),
        n_registers=4,
        gating=True,
        frozen_stages=-1,
        init_cfg=dict(
            type="Pretrained",
            checkpoint=pretrained,
        ),
    ),
    decode_head=None,
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

# ---------------------------------------------------------------------------
# Runtime (minimal stubs required by MMSeg config loading)
# ---------------------------------------------------------------------------
default_scope = "mmseg"
log_level = "INFO"
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)
