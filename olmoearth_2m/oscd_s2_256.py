# custom_imports = dict(
#     imports=['projects.olmoearth.opencd_olmoearth'],
#     allow_failed_imports=False)

dataset_type = 'OSCD_Dataset'
data_root = '/mnt/ht2-nas2/EO_test/wry/Copernicus/Data/OSCD/'
crop_size = (256, 256)
meta_keys = (
    'img_path', 'seg_map_path', 'seg_map_path_from', 'seg_map_path_to',
    'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip',
    'flip_direction', 'sample_id')

s2_band_stats = dict(
    mean=[
        1117.2,  # B02
        1041.8,  # B03
        946.5,   # B04
        2301.2,  # B08
        1199.1,  # B05
        2003.0,  # B06
        2374.0,  # B07
        2599.7,  # B8A
        1820.6,  # B11
        1118.2,  # B12
        1353.7,  # B01
        732.1,   # B09
    ],
    std=[
        736.0,   # B02
        684.8,   # B03
        620.0,   # B04
        1545.5,  # B08
        791.9,   # B05
        1341.3,  # B06
        1595.4,  # B07
        1750.1,  # B8A
        1216.5,  # B11
        736.7,   # B12
        897.3,   # B01
        475.1,   # B09
    ],
)

train_pipeline = [
    dict(type='MultiImgLoadGeoTiffImageFromFile'),
    dict(type='MultiImgLoadOSCDAnnotations'),
    dict(
        type='MultiImgNormalizeMultibandImage',
        mean=s2_band_stats['mean'],
        std=s2_band_stats['std']),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='horizontal'),
    dict(type='MultiImgRandomFlip', prob=0.5, direction='vertical'),
    dict(type='MultiImgPackSegInputs')
]
test_pipeline = [
    dict(type='MultiImgLoadGeoTiffImageFromFile'),
    dict(
        type='MultiImgNormalizeMultibandImage',
        mean=s2_band_stats['mean'],
        std=s2_band_stats['std']),
    dict(type='MultiImgLoadOSCDAnnotations'),
    dict(type='MultiImgPackSegInputs', meta_keys=meta_keys)
]

train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=10,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            official_format=True,
            split='train',
            pipeline=train_pipeline)))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        official_format=True,
        split='test',
        pipeline=test_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        official_format=True,
        split='test',
        pipeline=test_pipeline))

val_evaluator = dict(type='mmseg.IoUMetric', iou_metrics=['mFscore', 'mIoU'])
test_evaluator = val_evaluator
