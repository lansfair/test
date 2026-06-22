_base_ = [
    './olmoearth_2m_upernet_OSCD.py',
    './oscd_s2_256.py',
    './default_runtime.py'
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
log_processor = dict(by_epoch=True)
randomness = dict(seed=0)
work_dir = "/mnt/qh2-nas3/EO_test/wry/work_dirs/olmoearth-2m-base_upernet_1xb16-50e_oscd-s2-256x256"

# Replace the inherited backbone as a whole so no fields from another
# OLMoEarth backbone configuration are merged into the RGB adapter.
model = dict(
    backbone=dict(
        _delete_=True,
        type="OlmoEarth2mRGBBackbone",
        model_config_path=(
            "/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/"
            "new_olmoearth/olmoearth_2m/weight/config.json"
        ),
        init_cfg=dict(
            type="Pretrained",
            checkpoint=(
                "/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/"
                "new_olmoearth/olmoearth_2m/weight/weights.pth"
            ),
        ),
        modality="rgb",
        patch_size=4,
        num_timesteps=1,
        out_channels=768,
        pooling_type="mean",
        raw_bands=12,
        proj_target_bands=4,
        out_indices=(0, 1, 2, 3),
    )
)

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=1e-3, weight_decay=0.01))

param_scheduler = [
    dict(
        type='OneCycleLR',
        eta_max=1e-3,
        pct_start=0.0,
        anneal_strategy='cos',
        begin=0,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True)
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best='mIoU',
        rule='greater',
        save_last=True),
    logger=dict(type='LoggerHook', interval=10, log_metric_by_epoch=True),
    visualization=dict(
        type='OlmoEarthOSCDVisualizationHook',
        interval=1,
        draw=True,
        show=False))
