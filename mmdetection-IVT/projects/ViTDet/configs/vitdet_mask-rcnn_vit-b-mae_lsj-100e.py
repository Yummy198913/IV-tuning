_base_ = [
    '../../../configs/_base_/models/mask-rcnn_r50_fpn.py',
    './lsj-100e_msrs.py',
]

custom_imports = dict(imports=['projects.ViTDet.vitdet'])

backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='LN2d', requires_grad=True)
# image_size = (1024, 1024)
image_size = (512, 512)
batch_augments = [
    dict(type='BatchFixedSizePad_vi_ir', size=image_size, pad_mask=True)
]

# model settings
model = dict(
    data_preprocessor=dict(pad_size_divisor=32, batch_augments=batch_augments),
    backbone_vi=dict(
        act_cfg=dict(type='GELU'),
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        embed_dims=1024,
        img_size=(
            512,
            512,
        ),
        in_channels=3,
        init_values=1.0,
        mlp_ratio=4,
        norm_cfg=dict(eps=1e-06, type='LN'),
        norm_eval=False,
        num_heads=16,
        num_layers=24,
        out_indices=[
            9,
            14,
            19,
            23,
        ],
        patch_size=16,
        init_cfg=dict(
            checkpoint=
            '/raid/zhangyaming2/Projects/mmsegmentation/pretrained/mae_pretrain_vit_large.pth',
            type='Pretrained'),
        type='ViT'),
    neck=dict(
        _delete_=True,
        type='SimpleFPN',
        backbone_channel=1024,
        in_channels=[192, 384, 768, 768],
        out_channels=256,
        num_outs=5,
        norm_cfg=norm_cfg),
    rpn_head=dict(num_convs=2),
    roi_head=dict(
        bbox_head=dict(
            type='Shared4Conv1FCBBoxHead',
            conv_out_channels=256,
            norm_cfg=norm_cfg),
        mask_head=dict(norm_cfg=norm_cfg)))

custom_hooks = [dict(type='Fp16CompresssionHook')]
