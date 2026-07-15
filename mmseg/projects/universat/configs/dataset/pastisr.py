"""Example PASTIS-R dataset config for UniverSat.

Replace ``data_root``, ``split`` paths, ``mean``/``std`` values and
``ignore_index``/``num_classes`` with the settings of your actual dataset.
"""

import os

# ---------------------------------------------------------------------------
# Dataset layout
# ---------------------------------------------------------------------------
dataset_type = 'UniverSatSegDataset'
data_root = os.path.join(os.environ.get('MM_ARCHIVE_DATA_HOME', 'data'), 'pastisr')

modalities = ['s2', 's1']
crop_size = (360, 360)
num_classes = 18
ignore_index = 0  # placeholder; PASTIS-R uses 0=background (valid) and 19=void (ignore).

# Placeholder normalization. Compute real statistics from your training split
# (e.g. via src/data/utils.estimate_norm) and paste the values here.
norm_cfg = dict(
    mean={
        's2': [0.0] * 10,
        's1': [0.0] * 3,
    },
    std={
        's2': [1.0] * 10,
        's1': [1.0] * 3,
    },
)

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadMultimodalFromFile', modalities=modalities),
    dict(type='LoadAnnotations'),
    dict(type='NormalizeMultimodal', **norm_cfg),
    dict(type='PackUniverSatInputs'),
]

test_pipeline = [
    dict(type='LoadMultimodalFromFile', modalities=modalities),
    dict(type='LoadAnnotations'),
    dict(type='NormalizeMultimodal', **norm_cfg),
    dict(type='PackUniverSatInputs'),
]

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        split='splits/train.json',
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        split='splits/val.json',
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mFscore'],
    ignore_index=ignore_index,
)
test_evaluator = val_evaluator
