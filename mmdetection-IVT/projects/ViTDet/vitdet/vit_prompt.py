# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmengine.logging import MMLogger
from mmengine.model import BaseModule
from mmengine.runner.checkpoint import CheckpointLoader

from mmdet.registry import MODELS
from zym_tools.Write2Log import write2log
from zym_tools.feature_token import token2feature, feature2token
from zym_tools.simam import SimAM


class LN2d(nn.Module):
    """A LayerNorm variant, popularized by Transformers, that performs
    pointwise mean and variance normalization over the channel dimension for
    inputs that have shape (batch_size, channels, height, width)."""

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


def get_abs_pos(abs_pos, has_cls_token, hw):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    assert size * size == xy_num

    if size != h or size != w:
        new_abs_pos = F.interpolate(
            abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2),
            size=(h, w),
            mode='bicubic',
            align_corners=False,
        )

        return new_abs_pos.permute(0, 2, 3, 1)
    else:
        return abs_pos.reshape(1, h, w, -1)


def get_rel_pos(q_size, k_size, rel_pos):
    """
    Get relative positional embeddings according to the relative positions
    of query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode='linear',
        )
        rel_pos_resized = rel_pos_resized.reshape(-1,
                                                  max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords -
                       k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]


def add_decomposed_rel_pos(attn, q, rel_pos_h, rel_pos_w, q_size, k_size):
    """
    Args:
        attn (Tensor): attention map.
        q (Tensor):
            query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor):
            relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor):
            relative position embeddings (Lw, C) for width axis.
        q_size (Tuple):
            spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple):
            spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum('bhwc,hkc->bhwk', r_q, Rh)
    rel_w = torch.einsum('bhwc,wkc->bhwk', r_q, Rw)

    attn = (attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] +
            rel_w[:, :, :, None, :]).view(B, q_h * q_w, k_h * k_w)

    return attn


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size,
               window_size, C)
    windows = x.permute(0, 1, 3, 2, 4,
                        5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size,
                     window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=True,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 input_size=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            # initialize relative positional embeddings
            self.rel_pos_h = nn.Parameter(
                torch.zeros(2 * input_size[0] - 1, head_dim))
            self.rel_pos_w = nn.Parameter(
                torch.zeros(2 * input_size[1] - 1, head_dim))

            if not rel_pos_zero_init:
                nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
                nn.init.trunc_normal_(self.rel_pos_w, std=0.02)

    def forward(self, x):
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads,
                                  -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.num_heads, H * W, -1).unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)

        if self.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.rel_pos_h,
                                          self.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W,
                            -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.proj(x)

        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_cfg=dict(type='GELU'),
            bias=True,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_activation_layer(act_cfg)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        norm_cfg=dict(type='LN', eps=1e-6),
        act_cfg=dict(type='GELU'),
        use_rel_pos=False,
        rel_pos_zero_init=True,
        window_size=0,
        input_size=None,
    ):
        super().__init__()
        self.norm1 = build_norm_layer(norm_cfg, dim)[1]
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            use_rel_pos=use_rel_pos,
            rel_pos_zero_init=rel_pos_zero_init,
            input_size=input_size if window_size == 0 else
            (window_size, window_size),
        )

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_cfg=act_cfg)

        self.window_size = window_size

        # fixme
        self.mona_prompt1 = Mona(dim, 64)
        self.mona_prompt2 = Mona(dim, 64)
        self.extra_prompt1 = ExtraAdapter(dim, 8)
        self.extra_prompt2 = ExtraAdapter(dim, 8)

    def forward(self, x, x1):
        shortcut = x  # 4, 32, 32, 768 bhwc
        x = self.norm1(x)
        # Window partition
        if self.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)

        x = self.attn(x)
        # Reverse window partition
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (H, W))

        x = shortcut + self.drop_path(x)

        '''internal adapter with prompts'''
        # bhwc
        x = self.mona_prompt1(x)
        x = x + self.extra_prompt1(x1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = self.mona_prompt2(x)
        x = x + self.extra_prompt2(x1)

        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(self,
                 kernel_size=(16, 16),
                 stride=(16, 16),
                 padding=(0, 0),
                 in_chans=3,
                 embed_dim=768):
        """
        Args:
            kernel_size (Tuple): kernel size of the projection layer.
            stride (Tuple): stride of the projection layer.
            padding (Tuple): padding size of the projection layer.
            in_chans (int): Number of input image channels.
            embed_dim (int):  embed_dim (int): Patch embedding dimension.
        """
        super().__init__()

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding)

    def forward(self, x):
        x = self.proj(x)
        # B C H W -> B H W C
        x = x.permute(0, 2, 3, 1)
        return x

# fixme

# class MonaOp(nn.Module):
#     def __init__(self, in_features):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=3 // 2, groups=in_features)
#         self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=5 // 2, groups=in_features)
#         self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=7, padding=7 // 2, groups=in_features)
#
#         self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )
#
#     def forward(self, x):
#         identity = x
#         conv1_x = self.conv1(x)
#         conv2_x = self.conv2(x)
#         conv3_x = self.conv3(x)
#
#         x = (conv1_x + conv2_x + conv3_x) / 3.0 + identity
#
#         identity = x
#
#         x = self.projector(x)
#
#         return identity + x

class PConv(nn.Module):
    def __init__(self, in_features, n_div=4):
        super().__init__()
        self.dim_conv3 = in_features // n_div
        self.dim_untouched = in_features - self.dim_conv3
        self.partial_conv = nn.Conv2d(self.dim_conv3, self.dim_conv3,
                                      kernel_size=3, padding=3 // 2, groups=self.dim_conv3)
        self.project1 = nn.Conv2d(in_features, in_features, kernel_size=1)
        self.BN = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU(inplace=True)
        self.project2 = nn.Conv2d(in_features, in_features, kernel_size=1)

    def forward(self, x):
        identity = x
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv(x1)
        x = torch.cat([x1, x2], dim=1) + identity

        identity = x
        x = self.project1(x)
        x = self.relu(self.BN(x))
        x = self.project2(x) + identity
        return x


class Mona(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim):
        super().__init__()

        self.project1 = nn.Linear(in_dim, hidden_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(hidden_dim, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = PConv(hidden_dim)

        self.norm = nn.LayerNorm(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim))
        self.beta = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        # bhwc
        identity = x

        x = self.norm(x) * self.gamma + self.beta

        project1 = self.project1(x)

        # b, n, c = project1.shape
        # h, w = hw_shapes
        # b, h, w, c = project1.shape
        # project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = project1.permute(0, 3, 1, 2).contiguous()  # b, c, h, w
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).contiguous()  # b, h, w, c
        # project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)
        project2 = project2

        return identity + project2


# fixme
class ExtraAdapter(nn.Module):
    def __init__(self, in_dim, hide_dim, ):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, hide_dim)
        self.up = nn.Linear(hide_dim, in_dim)
        self.act = nn.GELU()
        self.act_prompt = nn.GELU()
        self.simam = SimAM()
        self.scale = nn.Parameter(torch.ones(in_dim))
        self.bias = nn.Parameter(torch.zeros(in_dim))
        self.gamma = nn.Parameter(torch.ones(in_dim))
        self.beta = nn.Parameter(torch.zeros(in_dim))

    def forward(self, x):
        # bhwc
        x = self.norm(x) * self.gamma + self.beta
        x = self.down(x)
        # b, n, c = x.shape
        # h, w = hw_shapes
        # x = x.reshape(b, w, h, c).permute(0, 3, 1, 2)  # bcwh
        x = x.permute(0, 3, 1, 2).contiguous()  # bchw
        x = self.simam(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # bhwc
        # x = x.permute(0, 2, 3, 1).reshape(b, n, c)
        x = self.up(x) * self.scale + self.bias
        return self.act_prompt(x)


class Prompt_block(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(Prompt_block, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.simam = SimAM(hide_channel)
        self.adapter_conv = PConv(hide_channel)

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C / 2), :, :].contiguous()  # VIS
        x0 = self.conv0_0(x0)
        x1 = x[:, int(C / 2):, :, :].contiguous()  # INF
        x1 = self.conv0_1(x1)
        x0 = self.adapter_conv(x0) + self.simam(x1)

        return self.conv1x1(x0)


@MODELS.register_module()
class ViT_prompt(BaseModule):
    """Vision Transformer with support for patch or hybrid CNN input stage."""

    def __init__(self,
                 img_size=1024,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 drop_path_rate=0.0,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 act_cfg=dict(type='GELU'),
                 use_abs_pos=True,
                 use_rel_pos=False,
                 rel_pos_zero_init=True,
                 window_size=0,
                 window_block_indexes=(0, 1, 3, 4, 6, 7, 9, 10),
                 pretrain_img_size=224,
                 pretrain_use_cls_token=True,
                 init_cfg=None,
                 out_indices=-1):

        super().__init__()
        self.pretrain_use_cls_token = pretrain_use_cls_token
        self.init_cfg = init_cfg

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim)
        self.patch_embed_prompt = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim)

        '''prompt parameters'''
        self.prompt_type = 'shaw'  # FIXME: choose the type of prompt
        if self.prompt_type in ['shaw', 'deep']:
            prompt_blocks = []
            block_nums = depth if self.prompt_type == 'deep' else 1
            for i in range(block_nums):
                prompt_blocks.append(Prompt_block(inplanes=embed_dim, hide_channel=64, smooth=True))
            self.prompt_blocks = nn.Sequential(*prompt_blocks)

            prompt_norms = []
            norm_layer = nn.LayerNorm
            for i in range(block_nums):
                prompt_norms.append(norm_layer(embed_dim))
            self.prompt_norms = nn.Sequential(*prompt_norms)


        if use_abs_pos:
            num_patches = (pretrain_img_size // patch_size) * (
                pretrain_img_size // patch_size)
            num_positions = (num_patches +
                             1) if pretrain_use_cls_token else num_patches
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_positions, embed_dim))
        else:
            self.pos_embed = None

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = depth - 1
                self.out_indices = out_indices
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of list, int or tuple')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i in window_block_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size))
            for i in range(depth)
        ])

        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def init_weights(self):
        logger = MMLogger.get_current_instance()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            self.apply(self._init_weights)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = CheckpointLoader.load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            self.load_state_dict(_state_dict, False)

        txt_path = '/data/hdd/zhangyaming/Projects/mmdetectionDM/zym_log/{}/param.txt'.format(os.path.basename((__file__).split('.')[0]))
        self.frozen_exclude = ['prompt']
        if 'all' in self.frozen_exclude:
            print('All params will be updated')
            all_parameters = sum(p.numel() for _, p in self.named_parameters())
            print('number of all params (M): %.2f' % (all_parameters / 1.e6))
            print('updated params / all params: 100%')
            return
        else:
            updated_param = []
            for name, param in self.named_parameters():
                for i in self.frozen_exclude:
                    if i in name:
                        updated_param.append(name)
            assert len(updated_param) != 0
            print('The following parameters will be updated:{}'.format(updated_param))
            write2log(txt_path, 'w', 'The following parameters will be updated:{}'.format(updated_param))
            for name, param in self.named_parameters():
                if name in updated_param:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            n_parameters = sum(p.numel() for _, p in self.named_parameters() if p.requires_grad)
            all_parameters = sum(p.numel() for _, p in self.named_parameters())
            print('number of params (M) to be updated: %.2f' % (n_parameters / 1.e6))
            print('number of all params (M): %.2f' % (all_parameters / 1.e6))
            print('updated params / all params: %.2f %%' % ((n_parameters / 1.e6) / (all_parameters / 1.e6) * 100))
            write2log(txt_path, 'a', 'number of params (M) to be updated: %.2f' % (n_parameters / 1.e6))
            write2log(txt_path, 'a', 'number of all params (M): %.2f' % (all_parameters / 1.e6))
            write2log(txt_path, 'a',
                      'updated params / all params: %.2f %%' % ((n_parameters / 1.e6) / (all_parameters / 1.e6) * 100))

    def forward(self, x, z):
        x = self.patch_embed(x)  # 4, 32, 32, 768
        z = self.patch_embed_prompt(z)

        '''prompt at input'''
        vis_feat = self.prompt_norms[0](x)
        inf_feat = self.prompt_norms[0](z)
        x_feat = torch.cat([vis_feat, inf_feat], dim=3)
        x_feat = x_feat.permute(0, 3, 2, 1).contiguous()  # bhwc -> bcwh
        x_feat = self.prompt_blocks[0](x_feat)
        x_prompted = x_feat.permute(0, 3, 2, 1).contiguous()  # bhwc
        x = x + x_prompted

        if self.pos_embed is not None:
            x = x + get_abs_pos(self.pos_embed, self.pretrain_use_cls_token,
                                (x.shape[1], x.shape[2]))

        outs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, x_prompted)
            x_prompted = x

            # if i in self.out_indices:
            #     out = x
            #     out = out.permute(0, 3, 1, 2)
            #     outs.append(out)
        x = x.permute(0, 3, 1, 2)


        # return tuple(outs)
        return x
