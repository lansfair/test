"""UniverSat Base + segmentation head on Potsdam (RGB-only).

Run from the MMSegmentation root (the folder that contains ``tools/``)::

    export PYTHONPATH=".:$PYTHONPATH"
    python tools/train.py \
        projects/universat/configs/potsdam_universat-base_seg.py \
        --work-dir work_dirs/potsdam_universat-base_seg
"""

import os

_base_ = [
    '../../../../configs/_base_/default_runtime.py',
    './dataset/potsdam.py',
]

custom_imports = dict(
    imports=['projects.universat.universat'],
    allow_failed_imports=False,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
backbone_embed_dim = 768

model = dict(
    type='EncoderDecoder',
    data_preprocessor=dict(type='UniverSatDataPreprocessor'),
    backbone=dict(
        type='UniverSatBackbone',
        modalities=['rgb'],
        embed_dim=backbone_embed_dim,
        num_heads=12,
        patch_size=40,
        output_grid=36,  # 512 x 512 / 40m ~ 12.8 -> use 36 (6x6) for Potsdam
        block_type=('Bi_ACA_in', 'SAx12', 'Bilinear_out', 'CA_Sub'),
        n_registers=4,
        gating=True,
        frozen_stages=-1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=os.path.join(
                os.environ.get('MM_ARCHIVE_CKPT_HOME', 'checkpoints'),
                'universat_base.safetensors',
            ),
        ),
    ),
    decode_head=dict(
        type='UniverSatSegHead',
        in_channels=backbone_embed_dim,
        in_index=0,
        channels=256,
        num_convs=2,
        output_size=crop_size,
        num_classes=num_classes,
        ignore_index=ignore_index,
        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
)

# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-4, weight_decay=0.05),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=1e-4,
        pct_start=0.0,
        anneal_strategy='cos',
        begin=0,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True,
    ),
]

log_processor = dict(by_epoch=True)

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'),
)
