auto_scale_lr = dict(base_batch_size=8, enable=False)
batch_size = 16
crop_size = (
    256,
    256,
)
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.dinov3.opencd_dinov3',
    ])
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size_divisor=16,
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ],
    test_cfg=dict(size_divisor=16),
    type='DualInputSegDataPreProcessor')
data_root = '/mnt/ht2-nas2/EO_test/dataset/ChangeDetection/WHU-CD/WHU-CD-SPLIT'
dataset_type = 'WHU_CD_Dataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, interval=4000, save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(
        img_shape=(
            256,
            256,
            3,
        ), interval=1, type='CDVisualizationHook'))
default_scope = 'opencd'
embed_dim = 1408
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.75,
    1.0,
    1.25,
]
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    auxiliary_head=dict(
        align_corners=False,
        channels=256,
        concat_input=False,
        dropout_ratio=0.1,
        ignore_index=255,
        in_channels=1408,
        in_index=2,
        loss_decode=dict(
            avg_non_ignore=True,
            loss_weight=0.4,
            type='mmseg.CrossEntropyLoss',
            use_sigmoid=False),
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        num_convs=1,
        type='mmseg.FCNHead'),
    backbone=dict(
        checkpoint=
        '/mnt/htzzb2/EO_test/wj1/swin_self_developed_weights/swintransformer-huge.pt',
        frozen=True,
        img_size=(
            256,
            256,
        ),
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_size=4,
        type='DINOv3DistilledSwinHuge',
        use_ema=True,
        window_size=8),
    backbone_inchannels=3,
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size_divisor=16,
        std=[
            58.395,
            57.12,
            57.375,
            58.395,
            57.12,
            57.375,
        ],
        test_cfg=dict(size_divisor=16),
        type='DualInputSegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=512,
        dropout_ratio=0.1,
        ignore_index=255,
        in_channels=[
            1408,
            1408,
            1408,
            1408,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                avg_non_ignore=True,
                loss_weight=1.0,
                type='mmseg.CrossEntropyLoss',
                use_sigmoid=False),
            dict(
                ignore_index=255,
                loss_weight=0.5,
                type='mmseg.DiceLoss',
                use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=2,
        pool_scales=(
            1,
            2,
            3,
            6,
        ),
        type='mmseg.UPerHead'),
    neck=dict(
        in_channels=[
            352,
            704,
            1408,
            2816,
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_inputs=1,
        out_channels=1408,
        policy='abs_diff',
        scales=[
            4,
            2,
            1,
            0.5,
        ],
        type='DINOv3FeatureFusionPyramid'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        256,
        256,
    ), mode='slide', stride=(
        128,
        128,
    )),
    train_cfg=dict(),
    type='SiamEncoderDecoder')
model_wrapper_cfg = dict(
    find_unused_parameters=True, type='MMDistributedDataParallel')
norm_cfg = dict(requires_grad=True, type='SyncBN')
num_classes = 2
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.01, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.001, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(backbone=dict(decay_mult=1.0, lr_mult=0.1)),
        norm_decay_mult=0.0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1000,
        by_epoch=False,
        end=40000,
        eta_min=1e-06,
        power=0.9,
        type='PolyLR'),
]
patch_size = 16
resume = False
swin_huge_channels = [
    352,
    704,
    1408,
    2816,
]
swin_huge_checkpoint = '/mnt/htzzb2/EO_test/wj1/swin_self_developed_weights/swintransformer-huge.pt'
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='test/T1',
            img_path_to='test/T2',
            seg_map_path='test/GT'),
        data_root=
        '/mnt/ht2-nas2/EO_test/dataset/ChangeDetection/WHU-CD/WHU-CD-SPLIT',
        img_suffix='.tif',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='MultiImgResize'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        seg_map_suffix='.tif',
        test_mode=True,
        type='WHU_CD_Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
test_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        256,
        256,
    ), type='MultiImgResize'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgPackSegInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=4000)
train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        data_prefix=dict(
            img_path_from='train/T1',
            img_path_to='train/T2',
            seg_map_path='train/GT'),
        data_root=
        '/mnt/ht2-nas2/EO_test/dataset/ChangeDetection/WHU-CD/WHU-CD-SPLIT',
        img_suffix='.tif',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(type='MultiImgLoadAnnotations'),
            dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
            dict(
                cat_max_ratio=0.75,
                crop_size=(
                    256,
                    256,
                ),
                type='MultiImgRandomCrop'),
            dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
            dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
            dict(
                brightness_delta=10,
                contrast_range=(
                    0.8,
                    1.2,
                ),
                hue_delta=10,
                saturation_range=(
                    0.8,
                    1.2,
                ),
                type='MultiImgPhotoMetricDistortion'),
            dict(type='MultiImgPackSegInputs'),
        ],
        seg_map_suffix='.tif',
        test_mode=False,
        type='WHU_CD_Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(degree=180, prob=0.5, type='MultiImgRandomRotate'),
    dict(
        cat_max_ratio=0.75, crop_size=(
            256,
            256,
        ), type='MultiImgRandomCrop'),
    dict(direction='horizontal', prob=0.5, type='MultiImgRandomFlip'),
    dict(direction='vertical', prob=0.5, type='MultiImgRandomFlip'),
    dict(
        brightness_delta=10,
        contrast_range=(
            0.8,
            1.2,
        ),
        hue_delta=10,
        saturation_range=(
            0.8,
            1.2,
        ),
        type='MultiImgPhotoMetricDistortion'),
    dict(type='MultiImgPackSegInputs'),
]
tta_model = dict(type='mmseg.SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='MultiImgLoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True, scale_factor=0.75, type='MultiImgResize'),
                dict(keep_ratio=True, scale_factor=1.0, type='MultiImgResize'),
                dict(
                    keep_ratio=True, scale_factor=1.25, type='MultiImgResize'),
            ],
            [
                dict(
                    direction='horizontal',
                    prob=0.0,
                    type='MultiImgRandomFlip'),
                dict(
                    direction='horizontal',
                    prob=1.0,
                    type='MultiImgRandomFlip'),
            ],
            [
                dict(type='MultiImgLoadAnnotations'),
            ],
            [
                dict(type='MultiImgPackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_prefix=dict(
            img_path_from='val/T1',
            img_path_to='val/T2',
            seg_map_path='val/GT'),
        data_root=
        '/mnt/ht2-nas2/EO_test/dataset/ChangeDetection/WHU-CD/WHU-CD-SPLIT',
        img_suffix='.tif',
        pipeline=[
            dict(type='MultiImgLoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                256,
                256,
            ), type='MultiImgResize'),
            dict(type='MultiImgLoadAnnotations'),
            dict(type='MultiImgPackSegInputs'),
        ],
        seg_map_suffix='.tif',
        test_mode=True,
        type='WHU_CD_Dataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mFscore',
        'mIoU',
    ], type='mmseg.IoUMetric')
vis_backends = [
    dict(type='CDLocalVisBackend'),
]
visualizer = dict(
    alpha=1.0,
    name='visualizer',
    type='CDLocalVisualizer',
    vis_backends=[
        dict(type='CDLocalVisBackend'),
    ])
work_dir = '/mnt/htzzb2/EO_test/wry/work_dirs/dinov3-distilled-swin-huge_upernet_4xb4-40k_whucd-256x256-frozen'
