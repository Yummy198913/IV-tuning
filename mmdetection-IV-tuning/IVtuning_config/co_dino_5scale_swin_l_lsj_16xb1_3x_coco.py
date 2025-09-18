auto_scale_lr = dict(base_batch_size=16)
backend_args = None
batch_augments = [
    dict(pad_mask=True, size=(
        640,
        640,
    ), type='BatchFixedSizePad_vi_ir'),
]
custom_imports = dict(
    allow_failed_imports=False, imports=[
        'projects.CO-DETR.codetr',
    ])
data_root = '/data/d1/zhangyaming/Datasets/M3FD_det/'
dataset_type = 'M3FDDataset_vi_ir'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        by_epoch=True,
        interval=1,
        save_best='coco/bbox_mAP',
        max_keep_ckpts=3, # fixme
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
image_size = (
    640,
    640,
)
launcher = 'none'
load_from = '/data/d1/zhangyaming/Projects/mmdetectionDM/pretrain/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth'
load_pipeline = [
    dict(type='LoadImageFromFile_vi_ir'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.1,
            2.0,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResize_vi_ir'),
    dict(
        allow_negative_crop=True,
        crop_size=(
            640,
            640,
        ),
        crop_type='absolute_range',
        recompute_bbox=True,
        type='RandomCrop_vi_ir'),
    dict(min_gt_bbox_wh=(
        0.01,
        0.01,
    ), type='FilterAnnotations'),
    dict(prob=0.5, type='RandomFlip_vi_ir'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad_vi_ir'),
]
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
loss_lambda = 2.0
max_epochs = 12
max_iters = 270000
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        convert_weights=True,
        depths=[
            2,
            2,
            18,
            2,
        ],
        drop_path_rate=0.5,
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
        # type='SwinTransformer',  # fixme
        # type='SwinTransformer_ivt',  # fixme
        type='SwinTransformer_mona',  # fixme
        # type='SwinTransformer_rein',  # fixme
        # type='SwinTransformer_readapter',  # fixme
        # type='SwinTransformer_prompt_simple',  # fixme
        # type='SwinTransformer_freeze',  # fixme
        # type='SwinTransformer_prefix',  # fixme
        # type='SwinTransformer_adapters_nlp',  # fixme
        # type='SwinTransformer_adptformer',  # fixme
        # type='SwinTransformer_biadaptformer',  # fixme
        # type='SwinTransformer_prompt',
        # type='SwinTransformer_prompt1',
        window_size=12,
        with_cp=False),
    bbox_head=[
        dict(
            anchor_generator=dict(
                octave_base_scale=8,
                ratios=[
                    1.0,
                ],
                scales_per_octave=1,
                strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                    128,
                ],
                type='AnchorGenerator'),
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    0.1,
                    0.1,
                    0.2,
                    0.2,
                ],
                type='DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            loss_bbox=dict(loss_weight=24.0, type='GIoULoss'),
            loss_centerness=dict(
                loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
            loss_cls=dict(
                alpha=0.25,
                gamma=2.0,
                loss_weight=12.0,
                type='FocalLoss',
                use_sigmoid=True),
            num_classes=6,
            stacked_convs=1,
            type='CoATSSHead'),
    ],
    data_preprocessor=dict(
        batch_augments=[
            dict(pad_mask=True, size=(
                640,
                640,
            ), type='BatchFixedSizePad_vi_ir'),
        ],
        bgr_to_rgb=True,
        mean_vi=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        std_vi=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor_vi_ir'),
    eval_module='detr',
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
    query_head=dict(
        as_two_stage=True,
        dn_cfg=dict(
            box_noise_scale=1.0,
            group_cfg=dict(dynamic=True, num_dn_queries=100, num_groups=None),
            label_noise_scale=0.5),
        in_channels=2048,
        loss_bbox=dict(loss_weight=5.0, type='L1Loss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        loss_iou=dict(loss_weight=2.0, type='GIoULoss'),
        num_classes=6,
        num_query=900,
        positional_encoding=dict(
            normalize=True,
            num_feats=128,
            temperature=20,
            type='SinePositionalEncoding'),
        transformer=dict(
            decoder=dict(
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_heads=8,
                            type='MultiheadAttention'),
                        dict(
                            dropout=0.0,
                            embed_dims=256,
                            num_levels=5,
                            type='MultiScaleDeformableAttention'),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'cross_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='DetrTransformerDecoderLayer'),
                type='DinoTransformerDecoder'),
            encoder=dict(
                num_layers=6,
                transformerlayers=dict(
                    attn_cfgs=dict(
                        dropout=0.0,
                        embed_dims=256,
                        num_levels=5,
                        type='MultiScaleDeformableAttention'),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        'self_attn',
                        'norm',
                        'ffn',
                        'norm',
                    ),
                    type='BaseTransformerLayer'),
                type='DetrTransformerEncoder',
                with_cp=6),
            num_co_heads=2,
            num_feature_levels=5,
            type='CoDinoTransformer',
            with_coord_feat=False),
        type='CoDINOHead'),
    roi_head=[
        dict(
            bbox_head=dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=120.0, type='GIoULoss'),
                loss_cls=dict(
                    loss_weight=12.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=6,
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            bbox_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                ],
                finest_scale=56,
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            type='CoStandardRoIHead'),
    ],
    rpn_head=dict(
        anchor_generator=dict(
            octave_base_scale=4,
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                4,
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(loss_weight=12.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=12.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    test_cfg=[
        dict(max_per_img=300, nms=dict(iou_threshold=0.8, type='soft_nms')),
        dict(
            rcnn=dict(
                max_per_img=100,
                nms=dict(iou_threshold=0.5, type='nms'),
                score_thr=0.0),
            rpn=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=1000)),
        dict(
            max_per_img=100,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.6, type='nms'),
            nms_pre=1000,
            score_thr=0.0),
    ],
    train_cfg=[
        dict(
            assigner=dict(
                match_costs=[
                    dict(type='FocalLossCost', weight=2.0),
                    dict(box_format='xywh', type='BBoxL1Cost', weight=5.0),
                    dict(iou_mode='giou', type='IoUCost', weight=2.0),
                ],
                type='HungarianAssigner')),
        dict(
            rcnn=dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=False,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            rpn=dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.3,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=256,
                    pos_fraction=0.5,
                    type='RandomSampler')),
            rpn_proposal=dict(
                max_per_img=1000,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=4000)),
        dict(
            allowed_border=-1,
            assigner=dict(topk=9, type='ATSSAssigner'),
            debug=False,
            pos_weight=-1),
    ],
    type='CoDETR_vi_ir',
    use_lsj=True)
num_classes = 6
num_dec_layer = 6
optim_wrapper = dict(
    clip_grad=dict(max_norm=0.1, norm_type=2),
    optimizer=dict(lr=0.0002, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(custom_keys=dict(backbone=dict(lr_mult=0.1))),
    type='OptimWrapper')
param_scheduler = [
    dict(milestones=[
        30,
    ], type='MultiStepLR'),
]
# pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'
resume = False
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations_coco/instances_val2017_with_gt_masks.json',
        backend_args=None,
        data_prefix=dict(img_vi='test/vi/', img_ir='test/ir'),
        data_root='/data/d1/zhangyaming/Datasets/M3FD_det/',
        pipeline=[
            dict(type='LoadImageFromFile_vi_ir'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize_vi_ir'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file='/data/d1/zhangyaming/Datasets/M3FD_det/annotations_coco/instances_val2017_with_gt_masks.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(type='LoadImageFromFile_vi_ir'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize_vi_ir'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
train_cfg = dict(max_epochs=50,
                 type='EpochBasedTrainLoop',
                 val_interval=1,
                 )
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        _scope_='mmdet',
        dataset=dict(
            ann_file='annotations_coco/instances_train2017_with_gt_masks.json',
            backend_args=None,
            data_prefix=dict(img_vi='train/vi/', img_ir='train/ir'),
            data_root='/data/d1/zhangyaming/Datasets/M3FD_det/',
            filter_cfg=dict(filter_empty_gt=False, min_size=32),
            pipeline=[
                dict(type='LoadImageFromFile_vi_ir'),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    keep_ratio=True,
                    ratio_range=(
                        0.1,
                        2.0,
                    ),
                    scale=(
                        640,
                        640,
                    ),
                    type='RandomResize_vi_ir'),
                dict(
                    allow_negative_crop=True,
                    crop_size=(
                        640,
                        640,
                    ),
                    crop_type='absolute_range',
                    recompute_bbox=True,
                    type='RandomCrop_vi_ir'),
                dict(min_gt_bbox_wh=(
                    0.01,
                    0.01,
                ), type='FilterAnnotations'),
                dict(prob=0.5, type='RandomFlip_vi_ir'),
                dict(
                    pad_val=dict(img=(
                        114,
                        114,
                        114,
                    )),
                    size=(
                        640,
                        640,
                    ),
                    type='Pad_vi_ir'),
            ],
            type='M3FDDataset_vi_ir'),
        pipeline=[
            dict(max_num_pasted=100, type='CopyPaste_vi_ir'),
            dict(type='PackDetInputs_vi_ir'),
        ],
        type='MultiImageMixDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(max_num_pasted=100, type='CopyPaste_vi_ir'),
    dict(type='PackDetInputs_vi_ir'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        ann_file='annotations_coco/instances_val2017_with_gt_masks.json',
        backend_args=None,
        data_prefix=dict(img_vi='test/vi/', img_ir='test/ir'),
        data_root='/data/d1/zhangyaming/Datasets/M3FD_det/',
        pipeline=[
            dict(type='LoadImageFromFile_vi_ir'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize_vi_ir'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad_vi_ir'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
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
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file='/data/d1/zhangyaming/Datasets/M3FD_det/annotations_coco/instances_val2017_with_gt_masks.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    classwise=True,
    type='CocoMetric')
vis_backends = [
    dict(_scope_='mmdet', type='LocalVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
