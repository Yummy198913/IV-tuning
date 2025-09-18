auto_scale_lr = dict(base_batch_size=16)
backend_args = None
data_root = '/data/d1/zhangyaming/Datasets/M3FD_det/'
dataset_type = 'M3FDDataset_vi_ir'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook', max_keep_ckpts=3),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = '/data/d1/zhangyaming/Projects/mmdetectionDM/pretrain/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 36
model = dict(
    as_two_stage=True,
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.2,
        drop_rate=0.0,
        embed_dims=192,
        init_cfg=dict(
            checkpoint=
            '/data/d1/zhangyaming/Projects/mmdetectionDM/pretrain/swin_large_patch4_window12_384_22k.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            6,
            12,
            24,
            48,
        ],
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_norm=True,
        pretrain_img_size=384,
        qk_scale=None,
        qkv_bias=True,
        # type='SwinTransformer',
        # type='SwinTransformer_prompt',
        # type='SwinTransformer_rein',
        type='SwinTransformer_freeze',
        # type='SwinTransformer_biadaptformer',
        # type='SwinTransformer_adptformer',
        window_size=12,
        with_cp=True),
    bbox_head=dict(
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=6,
        sync_cls_avg_factor=True,
        type='DINOHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean_vi=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std_vi=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor_vi_ir'),
    decoder=dict(
        layer_cfg=dict(
            cross_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5),
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_heads=8)),
        num_layers=6,
        post_norm_cfg=None,
        return_intermediate=True),
    dn_cfg=dict(
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
        label_noise_scale=0.5),
    encoder=dict(
        layer_cfg=dict(
            ffn_cfg=dict(
                embed_dims=256, feedforward_channels=2048, ffn_drop=0.0),
            self_attn_cfg=dict(dropout=0.0, embed_dims=256, num_levels=5)),
        num_layers=6),
    neck=dict(
        act_cfg=None,
        in_channels=[
            192,
            384,
            768,
            1536,
        ],
        kernel_size=1,
        norm_cfg=dict(num_groups=32, type='GN'),
        num_outs=5,
        out_channels=256,
        type='ChannelMapper'),
    num_feature_levels=5,
    num_queries=900,
    positional_encoding=dict(
        normalize=True, num_feats=128, offset=0.0, temperature=20),
    test_cfg=dict(max_per_img=300),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                dict(iou_mode='giou', type='IoUCost', weight=2.0),
            ],
            type='HungarianAssigner')),
    type='DINO',
    with_box_refine=True)
num_levels = 5
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0001, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=36,
        gamma=0.1,
        milestones=[
            27,
            33,
        ],
        type='MultiStepLR'),
]
pretrained = '/data/hdd/zhangyaming/Projects/mmdetection/pretrain/swin_large_patch4_window12_384_22k.pth'
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations_coco/instances_val2017_with_gt_masks.json',
        backend_args=None,
        data_prefix=dict(img_ir='test/ir/', img_vi='test/vi/'),
        data_root='/data/d1/zhangyaming/Datasets/M3FD_det/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile_vi_ir'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize_vi_ir'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs_vi_ir'),
        ],
        test_mode=True,
        type='M3FDDataset_vi_ir'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file=
    '/data/d1/zhangyaming/Datasets/M3FD_det/annotations_coco/instances_val2017_with_gt_masks.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile_vi_ir'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize_vi_ir'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs_vi_ir'),
]
train_cfg = dict(max_epochs=50, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations_coco/instances_train2017_with_gt_masks.json',
        backend_args=None,
        data_prefix=dict(img_ir='train/ir/', img_vi='train/vi/'),
        data_root='/data/d1/zhangyaming/Datasets/M3FD_det/',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile_vi_ir'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(prob=0.5, type='RandomFlip_vi_ir'),
            dict(
                transforms=[
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    640,
                                ),
                                (
                                    512,
                                    640,
                                ),
                                (
                                    544,
                                    640,
                                ),
                                (
                                    576,
                                    640,
                                ),
                                (
                                    608,
                                    640,
                                ),
                                (
                                    640,
                                    640,
                                ),
                                (
                                    672,
                                    640,
                                ),
                                (
                                    704,
                                    640,
                                ),
                                (
                                    736,
                                    640,
                                ),
                                (
                                    768,
                                    640,
                                ),
                                (
                                    800,
                                    640,
                                ),
                            ],
                            type='RandomChoiceResize_vi_ir'),
                    ],
                    [
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    400,
                                    4200,
                                ),
                                (
                                    500,
                                    4200,
                                ),
                                (
                                    600,
                                    4200,
                                ),
                            ],
                            type='RandomChoiceResize_vi_ir'),
                        dict(
                            allow_negative_crop=True,
                            crop_size=(
                                384,
                                600,
                            ),
                            crop_type='absolute_range',
                            type='RandomCrop_vi_ir'),
                        dict(
                            keep_ratio=True,
                            scales=[
                                (
                                    480,
                                    640,
                                ),
                                (
                                    512,
                                    640,
                                ),
                                (
                                    544,
                                    640,
                                ),
                                (
                                    576,
                                    640,
                                ),
                                (
                                    608,
                                    640,
                                ),
                                (
                                    640,
                                    640,
                                ),
                                (
                                    672,
                                    640,
                                ),
                                (
                                    704,
                                    640,
                                ),
                                (
                                    736,
                                    640,
                                ),
                                (
                                    768,
                                    640,
                                ),
                                (
                                    800,
                                    640,
                                ),
                            ],
                            type='RandomChoiceResize_vi_ir'),
                    ],
                ],
                type='RandomChoice'),
            dict(type='PackDetInputs_vi_ir'),
        ],
        type='M3FDDataset_vi_ir'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile_vi_ir'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(prob=0.5, type='RandomFlip_vi_ir'),
    dict(
        transforms=[
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            640,
                        ),
                        (
                            512,
                            640,
                        ),
                        (
                            544,
                            640,
                        ),
                        (
                            576,
                            640,
                        ),
                        (
                            608,
                            640,
                        ),
                        (
                            640,
                            640,
                        ),
                        (
                            672,
                            640,
                        ),
                        (
                            704,
                            640,
                        ),
                        (
                            736,
                            640,
                        ),
                        (
                            768,
                            640,
                        ),
                        (
                            800,
                            640,
                        ),
                    ],
                    type='RandomChoiceResize_vi_ir'),
            ],
            [
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            400,
                            4200,
                        ),
                        (
                            500,
                            4200,
                        ),
                        (
                            600,
                            4200,
                        ),
                    ],
                    type='RandomChoiceResize_vi_ir'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        384,
                        600,
                    ),
                    crop_type='absolute_range',
                    type='RandomCrop_vi_ir'),
                dict(
                    keep_ratio=True,
                    scales=[
                        (
                            480,
                            640,
                        ),
                        (
                            512,
                            640,
                        ),
                        (
                            544,
                            640,
                        ),
                        (
                            576,
                            640,
                        ),
                        (
                            608,
                            640,
                        ),
                        (
                            640,
                            640,
                        ),
                        (
                            672,
                            640,
                        ),
                        (
                            704,
                            640,
                        ),
                        (
                            736,
                            640,
                        ),
                        (
                            768,
                            640,
                        ),
                        (
                            800,
                            640,
                        ),
                    ],
                    type='RandomChoiceResize_vi_ir'),
            ],
        ],
        type='RandomChoice'),
    dict(type='PackDetInputs_vi_ir'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations_coco/instances_val2017_with_gt_masks.json',
        backend_args=None,
        data_prefix=dict(img_ir='test/ir/', img_vi='test/vi/'),
        data_root='/data/d1/zhangyaming/Datasets/M3FD_det/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile_vi_ir'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize_vi_ir'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs_vi_ir'),
        ],
        test_mode=True,
        type='M3FDDataset_vi_ir'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file=
    '/data/d1/zhangyaming/Datasets/M3FD_det/annotations_coco/instances_val2017_with_gt_masks.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    classwise=True,
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
