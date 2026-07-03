"""
MMRotate 1.x config:
DINOv3 ViT-L/16 + DINOv3 Adapter + FPN + Oriented R-CNN on DIOR-R.

该配置保留原 FCOS 实验的 backbone、数据集和短训练策略，新增：
1. Oriented RPN：由五层 FPN 产生旋转候选框；
2. Rotated RoIAlign：从 P4/P8/P16/P32 提取旋转 RoI 特征；
3. 两层全连接 bbox head：完成 20 类分类和旋转框精修。

NaN 修复（详见 dinov3_fcos_mine/docs/orcnn_nan_debug.md）：
1. FilterAnnotations：滤掉 DIOR-R 中零宽/零高的退化 ship 框，避免
   MidpointOffsetCoder/DeltaXYWHTRBBoxCoder 做 log(0)/除零 -> loss NaN；
2. RegularizeRotatedBoxes：qbox->rbox 依赖 cv2.minAreaRect，不同 OpenCV
   版本与几何增强会把角度带出 le90；几何变换完成后统一归一到 [-90°,90°)。
"""

# mmrotate 自带的 datasets 基础是 DIORDataset + VOC 风格 ImageSets 布局，
# 与本机 DOTA 格式（train/images、train/labelTxt、DOTADataset）不同，故不
# 沿用 _base_/datasets/dior*.py，而在此内联数据集定义（路径保持原值）。
_base_ = [
    # '/mnt/ht2-nas2/00-model/00-ds/mmlab/mmrotate/configs/_base_/datasets/dota.py',
    '/mnt/ht2-nas2/EO_test/xyz/Dinov3_ORCNN/mmlab/mmrotate/configs/_base_/schedules/schedule_1x.py',
    '/mnt/ht2-nas2/EO_test/xyz/Dinov3_ORCNN/mmlab/mmrotate/configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['dinov3_mmrotate0'], allow_failed_imports=False)

angle_version = 'le90'
num_classes = 20
backend_args = None

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel', find_unused_parameters=True)

model = dict(
    # Oriented R-CNN 是旋转版两阶段检测器，复用 Faster R-CNN 外壳。
    type='mmdet.FasterRCNN',
    data_preprocessor=dict(
        type='mmdet.DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
        boxtype2tensor=False),

    # 与原 FCOS 配置相同：冻结 DINOv3，只训练 Adapter、FPN 和检测头。
    backbone=dict(
        type='DINOv3Adapter0',
        arch='vit_large',
        patch_size=16,
        freeze_backbone=False,
        finetune_vit=True),

    # Adapter 输出 stride 4/8/16/32；FPN 再生成 stride 64 的 P5。
    neck=dict(
        type='mmdet.FPN',
        in_channels=[1024, 1024, 1024, 1024],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=5,
        relu_before_extra_convs=True),

    # 第一阶段：在五层 FPN 上生成 oriented proposals。
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='mmdet.AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64],
            use_box_type=True),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_version=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss',
            beta=1.0 / 9.0,
            loss_weight=1.0)),

    # 第二阶段：对旋转候选框做 RoIAlign、分类和旋转框回归。
    roi_head=dict(
        type='mmdet.StandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            # RPN 使用五层，RoI head 按标准 Oriented R-CNN 使用前四层。
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='mmdet.Shared2FCBBoxHead',
            predict_box_type='rbox',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=num_classes,
            reg_predictor_cfg=dict(type='mmdet.Linear'),
            cls_predictor_cfg=dict(type='mmdet.Linear'),
            bbox_coder=dict(
                type='DeltaXYWHTRBBoxCoder',
                angle_version=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(0.0, 0.0, 0.0, 0.0, 0.0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='mmdet.CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            loss_bbox=dict(
                type='mmdet.SmoothL1Loss',
                beta=1.0,
                loss_weight=1.0))),

    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1,
                iou_calculator=dict(type='RBbox2HBboxOverlaps2D')),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='mmdet.MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='mmdet.RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),

    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(type='nms_rotated', iou_threshold=0.1),
            max_per_img=2000)))

# =============================================================================
# 数据集: DIOR-R (本机路径, DOTA 格式标签 -> DOTADataset)
# =============================================================================
data_root = '/mnt/ht2-nas2/EO_test/openmmlab-archive/dat/DIOR-R/'

dior_classes = (
    'airplane', 'airport', 'baseballfield', 'basketballcourt', 'bridge',
    'chimney', 'Expressway-Service-area', 'Expressway-toll-station',
    'dam', 'golffield', 'groundtrackfield', 'harbor', 'overpass', 'ship',
    'stadium', 'storagetank', 'tenniscourt', 'trainstation', 'vehicle',
    'windmill')

train_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # Oriented R-CNN 的 bbox coder 会除以 GT 宽高并计算 log；过滤零宽/零高
    # 退化框（如 DIOR-R 的 ship 0×N），避免 log(0)/除零 -> loss_bbox NaN。
    # keep_empty=True：若过滤后该图无 GT，返回 None -> 训练时自动重新取样。
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=True),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='mmdet.RandomFlip',
        prob=0.75,
        direction=['horizontal', 'vertical', 'diagonal']),
    # 必须在 RandomFlip 之后：几何增强会再次把角度带出 le90；
    # qbox->rbox 的 cv2.minAreaRect 在不同 OpenCV 下也可能产生非规范角度。
    dict(type='RegularizeRotatedBoxes', pattern=angle_version),
    dict(type='mmdet.PackDetInputs'),
]
val_pipeline = [
    dict(type='mmdet.LoadImageFromFile', backend_args=backend_args),
    dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
    # 验证时先 resize 图像，再加载 GT，避免 GT 被重复缩放。
    dict(type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
    dict(type='ConvertBoxType', box_type_mapping=dict(gt_bboxes='rbox')),
    # 验证集也过滤退化 GT，避免旋转 IoU/编码过程出现非法数值。
    # keep_empty=False：测试模式不能靠重新取样跳过图片；坏框是唯一 GT 时
    # 保留空样本（仍参与评测），不丢弃图像。
    dict(
        type='mmdet.FilterAnnotations',
        min_gt_bbox_wh=(1, 1),
        keep_empty=False),
    dict(type='RegularizeRotatedBoxes', pattern=angle_version),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor')),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=None,
    dataset=dict(
        type='ConcatDataset',
        ignore_keys=['DATASET_TYPE'],
        datasets=[
            dict(
                type='DOTADataset',
                data_root=data_root,
                ann_file='train/labelTxt/',
                data_prefix=dict(img_path='train/images/'),
                img_suffix='jpg',
                metainfo=dict(classes=dior_classes),
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline),
            dict(
                type='DOTADataset',
                data_root=data_root,
                ann_file='val/labelTxt/',
                data_prefix=dict(img_path='val/images/'),
                img_suffix='jpg',
                metainfo=dict(classes=dior_classes),
                filter_cfg=dict(filter_empty_gt=True),
                pipeline=train_pipeline)
        ]))
val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTADataset',
        data_root=data_root,
        ann_file='test/labelTxt/',
        data_prefix=dict(img_path='test/images/'),
        img_suffix='jpg',
        metainfo=dict(classes=dior_classes),
        test_mode=True,
        pipeline=val_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='DOTAMetric', metric='mAP')
test_evaluator = val_evaluator

# 正式训练：100 epoch，每 4 epoch 验证一次（与 FCOS baseline 的评测频率一致）。
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=4)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=1e-4,
        weight_decay=0.05),
    clip_grad=dict(max_norm=35, norm_type=2))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0 / 3,
        by_epoch=False,
        begin=0,
        end=200),
    dict(
        type='MultiStepLR',
        begin=0,
        end=100,
        by_epoch=True,
        milestones=[60, 80],
        gamma=0.1),
]

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=20),
    checkpoint=dict(type='CheckpointHook', interval=4))