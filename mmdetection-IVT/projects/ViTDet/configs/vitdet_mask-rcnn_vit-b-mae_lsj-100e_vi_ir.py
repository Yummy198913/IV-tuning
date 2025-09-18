auto_scale_lr = dict(base_batch_size=64)
custom_hooks = [
    dict(type='Fp16CompresssionHook'),
]
custom_imports = dict(imports=[
    'projects.ViTDet.vitdet',
])
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=1000,  # save checkpoint per x interval
        max_keep_ckpts=5,
        save_best='auto',
        type='CheckpointHook'),
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
image_size = (
    800,
    800,
)
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=False, type='LogProcessor', window_size=50)
model = dict(
    backbone_vi=dict(
        type='ConvMAE',
        patch_size=[4, 2, 2],
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4],
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        img_size=[800, 200, 100],
        init_values=1.0,
        drop_path_rate=0.2,
        out_indices=[3, 5, 7, 11],
        # init_cfg=dict(checkpoint=None, type='Pretrained')
    ),
    backbone_ir=dict(
        type='ConvMAE',
        patch_size=[4, 2, 2],
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4],
        qkv_bias=True,
        use_abs_pos_emb=True,
        use_rel_pos_bias=True,
        img_size=[800, 200, 100],
        init_values=1.0,
        drop_path_rate=0.2,
        out_indices=[3, 5, 7, 11],
        # init_cfg=dict(checkpoint=None, type='Pretrained')
    ),
    data_preprocessor=dict(
        batch_augments=[
            dict(pad_mask=True, size=(
                800,
                800,
            ), type='BatchFixedSizePad_vi_ir'),
        ],
        bgr_to_rgb=True,
        mean_vi=[
            123.675,
            116.28,
            103.53,
        ],
        std_vi=[
            58.395,
            57.12,
            57.375,
        ],
        mean_ir=[
            108.375,
            108.375,
            108.375,
        ],
        std_ir=[
            51.0,
            51.0,
            51.0,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        type='DetDataPreprocessor_vi_ir'),
    neck=dict(
        in_channels=[
            256,
            384,
            768,
            768,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    roi_head=dict(
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
            conv_out_channels=256,
            fc_out_channels=1024,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=False),
            norm_cfg=dict(requires_grad=True, type='LN2d'),
            num_classes=6,
            reg_class_agnostic=False,
            roi_feat_size=7,
            type='Shared4Conv1FCBBoxHead'),
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        mask_head=dict(
            conv_out_channels=256,
            in_channels=256,
            loss_mask=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
            norm_cfg=dict(requires_grad=True, type='LN2d'),
            num_classes=6,
            num_convs=4,
            type='FCNMaskHead'),
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        type='StandardRoIHead'),
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
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
        loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        num_convs=2,
        type='RPNHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.05),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            mask_size=28,
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
            nms_pre=2000)),
    type='MaskRCNN_vi_ir')
norm_cfg = dict(requires_grad=True, type='LN2d')
optim_wrapper = dict(
    constructor='LayerDecayOptimizerConstructor',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.0001, type='AdamW', weight_decay=0.1),
    paramwise_cfg=dict(decay_rate=0.7, decay_type='layer_wise', num_layers=12),
    type='AmpOptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=250, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=False,
        end=184375,
        gamma=0.1,
        milestones=[
            163889,
            177546,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations_coco/instances_val2017_with_gt_masks.json',
        data_prefix=dict(img_vi='imgs/vi/', img_ir='imgs/ir'),
        data_root='/raid/liufangcen/data/M3FD-TO-LIU/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile_vi_ir'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='Resize_vi_ir'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    800,
                    800,
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
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='/raid/liufangcen/data/M3FD-TO-LIU/annotations_coco/instances_val2017_with_gt_masks.json',
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')

train_cfg = dict(
    dynamic_intervals=[
        (
            180001,
            184375,
        ),
    ],
    max_iters=184375,
    type='IterBasedTrainLoop',
    val_interval=5000)  # evaluate metrics per x interval
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations_coco/instances_train2017_with_gt_masks.json',
        data_prefix=dict(img_vi='imgs/vi/', img_ir='imgs/ir'),
        data_root='/raid/liufangcen/data/M3FD-TO-LIU/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile_vi_ir'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(prob=0.5, type='RandomFlip_vi_ir'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.1,
                    2.0,
                ),
                scale=(
                    800,
                    800,
                ),
                type='RandomResize_vi_ir'),
            dict(
                allow_negative_crop=True,
                crop_size=(
                    800,
                    800,
                ),
                crop_type='absolute_range',
                recompute_bbox=True,
                type='RandomCrop_vi_ir'),
            dict(min_gt_bbox_wh=(
                0.01,
                0.01,
            ), type='FilterAnnotations'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    800,
                    800,
                ),
                type='Pad_vi_ir'),
            dict(type='PackDetInputs_vi_ir'),
        ],
        type='M3FDDataset_vi_ir'),
    num_workers=8,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations_coco/instances_val2017_with_gt_masks.json',
        data_prefix=dict(img_vi='imgs/vi/', img_ir='imgs/ir'),
        data_root='/raid/liufangcen/data/M3FD-TO-LIU/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile_vi_ir'),
            dict(keep_ratio=True, scale=(
                800,
                800,
            ), type='Resize_vi_ir'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    800,
                    800,
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
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='/raid/liufangcen/data/M3FD-TO-LIU/annotations_coco/instances_val2017_with_gt_masks.json',
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = './work_dirs/vitdet_mask-rcnn_convmae_lsj-100e_vi_ir-add-scratch'