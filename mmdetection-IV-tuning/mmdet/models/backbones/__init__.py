# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .cspnext import CSPNeXt
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .swin_rein import SwinTransformer_rein
from .swin_vpt import SwinTransformer_vpt
from .swin_prefix import SwinTransformer_prefix
from .swin_freeze import SwinTransformer_freeze
from .swin_adapter_nlp import SwinTransformer_adapters_nlp
from .swin_prompt import SwinTransformer_prompt
from .swin_prompt_ori import SwinTransformer_prompt1
from .swin_adaptformer import SwinTransformer_adptformer
from .swin_bi_adaptformer import SwinTransformer_biadaptformer
from .trident_resnet import TridentResNet
from .mae import MAE
from .mae_prompt import MAE_prompt
# from .vit_codetr import ViT
# from .vit_codetr_prompt import ViT_codetr_prompt
from .swin_iv_tuning import SwinTransformer_ivt
from .swin_mona import SwinTransformer_mona
from .swin_readapter import SwinTransformer_readapter
from .swin_prompt_simple import SwinTransformer_prompt_simple

__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet', 'CSPNeXt', 'MAE', 'MAE_prompt',
     'SwinTransformer_prompt', 'SwinTransformer_adptformer', 'SwinTransformer_prompt1',
    'SwinTransformer_biadaptformer', 'SwinTransformer_freeze', 'SwinTransformer_vpt', 'SwinTransformer_adapters_nlp',
    'SwinTransformer_prefix', 'SwinTransformer_rein', 'SwinTransformer_ivt', 'SwinTransformer_mona', 'SwinTransformer_readapter',
    'SwinTransformer_prompt_simple'
]
