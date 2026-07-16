"""DIOR dataset config for UniverSat embedding extraction.

This config defines the standard DIOR horizontal-box layout and uses
``UniverSatDIORDataset`` to convert each image's bounding boxes into a
semantic mask. The mask is saved alongside the extracted embedding for
reference; it is not used to train a segmentation head here.

Expected DIOR layout::

    ${MM_ARCHIVE_DATA_HOME}/DIOR/
      JPEGImages/
        00001.jpg
        ...
      Annotations/
        00001.xml
        ...
      ImageSets/Main/
        train.txt
        val.txt
        test.txt

If your DIOR release uses ``JPEGImages-trainval`` / ``JPEGImages-test`` or
``Annotations/Oriented Bounding Boxes/``, override ``img_subdir`` and
``ann_subdir`` in the dataloader dataset config.
"""

# import os

# ---------------------------------------------------------------------------
# Dataset layout
# ---------------------------------------------------------------------------
dataset_type = 'UniverSatDIORDataset'
# data_root = os.path.join(os.environ.get('MM_ARCHIVE_DATA_HOME', 'data'), 'DIOR')
data_root = '/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/DIOR'

modalities = ['rgb']
crop_size = (800, 800)
num_classes = 20
ignore_index = 255  # reserved; background is 0

# DIOR RGB images are uint8 [0, 255]; normalize to [0, 1].
# Compute real statistics from the training split if you prefer ImageNet-style
# normalization; for backbone feature extraction [0, 1] is usually sufficient.
norm_cfg = dict(
    mean={'rgb': [0.0, 0.0, 0.0]},
    std={'rgb': [255.0, 255.0, 255.0]},
)

# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------
# ``sample_id`` is included in meta_keys so that the embedding extraction
# script uses the DIOR image id as the output directory name.
test_pipeline = [
    dict(type='LoadMultimodalFromFile', modalities=modalities),
    dict(type='NormalizeMultimodal', **norm_cfg),
    dict(
        type='PackUniverSatInputs',
        meta_keys=(
            'img_path',
            'seg_map_path',
            'ori_shape',
            'img_shape',
            'pad_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'sample_id',
        ),
    ),
]

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        split='ImageSets/Main/train.txt',
        img_subdir='JPEGImages',
        ann_subdir='Annotations',
        pipeline=test_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        split='ImageSets/Main/val.txt',
        img_subdir='JPEGImages',
        ann_subdir='Annotations',
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        modalities=modalities,
        data_root=data_root,
        split='ImageSets/Main/test.txt',
        img_subdir='JPEGImages',
        ann_subdir='Annotations',
        pipeline=test_pipeline,
    ),
)
