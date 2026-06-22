
custom_imports = dict(
    imports=['projects.olmoearthfm.models'],
    allow_failed_imports=False)

# model settings
backbone_inchannels = 12
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='DualInputSegDataPreProcessor',
    size=(256, 256),
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

#自研olmoearth-2m
olmoearth_model_dir = "/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/new_olmoearth/olmoearth_2m/weight"
model_config_path = f"{olmoearth_model_dir}/config.json"
olmoearth_fm_checkpoint = "/mnt/ht2-nas2/EO_test/openmmlab-archive/pretrained/new_olmoearth/olmoearth_2m/weight/weights.pth"
# s2_band_wavelengths = [
#     440, 490, 560, 665, 705, 740, 783, 842, 860, 1370, 1610, 2190
# ]
# s2_band_bandwidths = [20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 90, 180]
s2_band_wavelengths = [
    490,  # B02
    560,  # B03
    665,  # B04
    842,  # B08
    705,  # B05
    740,  # B06
    783,  # B07
    860,  # B8A
   1610,  # B11
   2190,  # B12
    440,  # B01
    940,  # B09
]

s2_band_bandwidths = [
    65,   # B02
    35,   # B03
    30,   # B04
   115,   # B08
    15,   # B05
    15,   # B06
    20,   # B07
    20,   # B8A
    90,   # B11
   180,   # B12
    20,   # B01
    20,   # B09
]
patch_area = (16 * 10 / 1000)**2

num_timesteps = 1
patch_size = 4


model = dict(
    type='SiamEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone_inchannels=backbone_inchannels,
    backbone=dict(
        type="OlmoEarth2mBackbone",
        model_config_path=model_config_path,
        init_cfg=dict(type="Pretrained", checkpoint=olmoearth_fm_checkpoint),
        modality="rgb",
        patch_size=patch_size,
        num_timesteps=num_timesteps,
        out_channels=768,
        pooling_type="mean",
        raw_bands=12,
        proj_target_bands=4,
    ),
    neck=dict(
        type='FeatureFusionPyramid',
        policy='abs_diff',
        embed_dim=768,
        rescales=[4, 2, 1, 0.5],
        norm_cfg=norm_cfg),
    decode_head=dict(
        type='mmseg.UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0)),
    auxiliary_head=dict(
        type='mmseg.FCNHead',
        in_channels=768,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='mmseg.CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=0.4)),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(256, 256), stride=(128, 128)))
