mm_archive_home = "/mnt/ht2-nas2/EO_test/openmmlab-archive"
data_root = f"{mm_archive_home}/dat/DIOR"
pretrained_dir = "/mnt/htzzb2/EO_test/00-zhumx/Working/checkpoints/universat-chpts/pretrained_chpts"
model_config_path = f"{pretrained_dir}/config.json"
weights_path = f"{pretrained_dir}/model.safetensors"

custom_imports = dict(
    imports=["projects.universat.universat", "projects.universat.dior_embed.universat"],
    allow_failed_imports=False,
)

_base_ = ['./dataset/dior.py']


ignore_index = 255
num_classes = 1
num_timesteps = 1
crop_size = (512, 512)
patch_size = 4
hidden_dim = 768

train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="GenerateDummySegMap"),
    dict(type="PadToDivisor", divisor=patch_size, img_pad_value=0, seg_pad_value=ignore_index),
    dict(
        type="DIORNormalize",
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        num_timesteps=num_timesteps,
        keep_raw_input=True,
    ),
    dict(type="PackOlmoEarthSegInputs"),
]

test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatDIORDataset",
        data_root=data_root,
        split="train",
        data_prefix=dict(img="Images/trainval"),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatDIORDataset",
        data_root=data_root,
        split="val",
        data_prefix=dict(img="Images/trainval"),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="UniverSatDIORDataset",
        data_root=data_root,
        split="test",
        data_prefix=dict(img="Images/test"),
        pipeline=test_pipeline,
        test_mode=True,
    ),
)

val_evaluator = dict(
    type="OlmoEarthIoUMetric",
    num_classes=num_classes,
    ignore_index=ignore_index,
    iou_metrics=["mIoU"],
    use_valid_mask=False,
)
test_evaluator = val_evaluator

data_preprocessor = dict(
    type="OlmoEarthSegDataPreProcessor",
    mean=None,
    std=None,
    bgr_to_rgb=False,
    pad_val=0,
    seg_pad_val=ignore_index,
    size=crop_size,
    test_cfg=dict(size_divisor=patch_size),
)

model = dict(
    type="OlmoEarthEncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="OlmoEarthBackbone",
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=weights_path),
        modality="sentinel2_l2a",
        patch_size=patch_size,
        num_timesteps=num_timesteps,
        out_channels=hidden_dim,
        pooling_type="mean",
    ),
    decode_head=dict(
        type="OlmoEarthPatchLinearHead",
        in_channels=hidden_dim,
        channels=hidden_dim,
        in_index=0,
        num_classes=num_classes,
        patch_size=patch_size,
        ignore_index=ignore_index,
        use_valid_mask=False,
        valid_mask_loss=False,
        align_corners=True,
        loss_decode=dict(
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
        ),
    ),
    auxiliary_head=None,
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

custom_hooks = [dict(type="FreezeBackboneUntilEpochHook", unfreeze_epoch=None)]

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=0.1, weight_decay=0.0),
)

param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=1e-3,
        begin=0,
        end=5,
        by_epoch=True,
    ),
    dict(
        type="CosineAnnealingLR",
        eta_min=1e-5,
        begin=5,
        end=50,
        T_max=45,
        by_epoch=True,
    ),
]

train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=50, val_interval=10)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=10,
        save_best="mIoU",
        rule="greater",
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="OlmoEarthVisualizationHook"),
)

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

default_scope = "mmseg"
log_level = "INFO"
load_from = None
resume = True
auto_scale_lr = dict(enable=False, base_batch_size=8)
