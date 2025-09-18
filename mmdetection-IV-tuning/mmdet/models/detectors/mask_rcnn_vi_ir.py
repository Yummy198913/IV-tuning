# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import ConfigDict

from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig
from .two_stage_vi_ir import TwoStageDetector_vi_ir


@MODELS.register_module()
class MaskRCNN_vi_ir(TwoStageDetector_vi_ir):
    """Implementation of `Mask R-CNN <https://arxiv.org/abs/1703.06870>`_"""

    def __init__(self,
                 backbone_vi: ConfigDict,
                 # backbone_ir: ConfigDict,
                 rpn_head: ConfigDict,
                 roi_head: ConfigDict,
                 train_cfg: ConfigDict,
                 test_cfg: ConfigDict,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone_vi=backbone_vi,
            # backbone_ir=backbone_ir,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg,
            data_preprocessor=data_preprocessor)
