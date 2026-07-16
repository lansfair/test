"""Extraction config for UniverSat Base on PASTIS-R.

This config is used by ``projects/universat/tools/extract_embeddings.py`` to
extract dense backbone embeddings for downstream offline probes.

Expected PASTIS-R layout::

    ${MM_ARCHIVE_DATA_HOME}/PASTIS-R/
      metadata.geojson
      DATA_S2/S2_{id}.npy
      DATA_S1A/S1A_{id}.npy
      ANNOTATIONS/TARGET_{id}.npy
      NORM_S2_patch.json
      NORM_S1_patch.json

Run from the MMSegmentation root::

    export PYTHONPATH=".:$PWD/projects/universat:$PWD/projects/universat/pastis:$PYTHONPATH"
    python projects/universat/tools/extract_embeddings.py \
        projects/universat/configs/extract_embeddings_pastisr_universat-base.py \
        --output-root work_dirs/universat_pastisr_embeddings \
        --splits train val test
"""

import os

custom_imports = dict(
    imports=["projects.universat.universat", "universat_pastis"],
    allow_failed_imports=False,
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
data_root = os.path.join(
    os.environ.get("MM_ARCHIVE_DATA_HOME", "data"), "PASTIS-R"
)
pretrained = os.path.join(
    os.environ.get("MM_ARCHIVE_CKPT_HOME", "checkpoints"),
    "universat_base.safetensors",
)

crop_size = (128, 128)
num_classes = 20
ignore_index = 19  # void class; background (0) is valid
modalities = ["s2", "s1"]

train_pipeline = [
    dict(type="PackUniverSatPASTISInputs", modalities=modalities),
]
test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type="universat_pastis_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatPASTISDataset",
        data_root=data_root,
        modalities=modalities,
        folds=[1, 2, 3],
        norm_path=data_root,
        temporal_dropout=0,
        pipeline=train_pipeline,
        test_mode=True,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type="universat_pastis_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatPASTISDataset",
        data_root=data_root,
        modalities=modalities,
        folds=[4],
        norm_path=data_root,
        temporal_dropout=0,
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    collate_fn=dict(type="universat_pastis_collate"),
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatPASTISDataset",
        data_root=data_root,
        modalities=modalities,
        folds=[5],
        norm_path=data_root,
        temporal_dropout=0,
        pipeline=test_pipeline,
        test_mode=True,
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
        output_grid=128,  # 128 x 128 output tokens for PASTIS-R 128 x 128 patches
        block_type=("Bi_ACA_in", "SAx12", "Bilinear_out", "CA_Sub"),
        n_registers=4,
        gating=True,
        frozen_stages=-1,  # freeze/ thaw does not matter for inference
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
