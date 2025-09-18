import torch
import torch.nn as nn
# from timm.models.layers import DropPath
import numpy as np
import torch.nn.functional as F
from models.SwinTransformers import SwinTransformer
from models.SwinTransformers_ivt import SwinTransformer_ivt
from models.VisionTransformers import VisionTransformer
from models.eva_02 import EVA2
from models.eva_02_ivt import EVA2_ivt
from models.mae import MAE
from models.mae_prompt import MAE_prompt


def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
            )

backbone_norm_cfg = dict(type="LN", requires_grad=True, eps=1e-6)

class MAECPNet(nn.Module):
    def __init__(self):
        super(MAECPNet, self).__init__()
        # self.backbone = MAE(
        self.backbone = MAE_prompt(
            img_size=(384, 384),
            patch_size=16,
            in_channels=3,
            embed_dims=1024,
            num_layers=24,
            num_heads=16,
            mlp_ratio=4,
            out_indices=(9, 14, 19, 23),
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            norm_cfg=dict(type='LN', eps=1e-6),
            act_cfg=dict(type='GELU'),
            norm_eval=False,
            init_values=0.1
        )
        self.FA_Block2 = Block(dim=384)  # 384
        self.FA_Block3 = Block(dim=192)  # 192
        self.FA_Block4 = Block(dim=96)  # 96
        self.predtrans2 = nn.Conv2d(192, 1, kernel_size=3, padding=1)  # 192
        self.predtrans3 = nn.Conv2d(384, 1, kernel_size=3, padding=1)  # 384
        self.predtrans4 = nn.Conv2d(768, 1, kernel_size=3, padding=1)  # 768
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 = nn.Sequential(  # 1536
            nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(  # 1536
            nn.Conv2d(in_channels=1792, out_channels=384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_3 = nn.Sequential(  # 768
            nn.Conv2d(in_channels=1408, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_4 = nn.Sequential(  # 384
            nn.Conv2d(in_channels=1216, out_channels=96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            self.upsample2
        )
        self.predict_layer_1 = nn.Sequential(  # 96
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x, z):
        x = self.backbone(x, z)
        # [8, 1024, 96, 96] [8, 1024, 48, 48] [8, 1024, 24, 24] [8, 1024, 12, 12]
        # [192, 96, 96], [384, 48, 48], [768, 24, 24], [1536, 12, 12]
        xf_1 = x[3]  # 8 1024 12 12   # 1536,12,12
        xf_2 = x[2]  # 8 1024 24 24   # 768,24,24
        xf_3 = x[1]  # 8 1024 48 48   # 384,48,48
        xf_4 = x[0]  # 8 1024 96 96   # 192,96,96

        df_f_1 = self.deconv_layer_1(xf_1)  # 8, 768, 24, 24

        xc_1_2 = torch.cat((df_f_1, xf_2), 1)  # 8, 1792, 24, 24
        df_f_2 = self.deconv_layer_2(xc_1_2)  # 8, 384, 48, 48
        df_f_2 = self.FA_Block2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)  # 8, 1408, 48, 48
        df_f_3 = self.deconv_layer_3(xc_1_3)  # 8, 192, 96, 96
        df_f_3 = self.FA_Block3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)  # 8, 1216, 96, 96
        df_f_4 = self.deconv_layer_4(xc_1_4)  # 8, 96, 192, 192
        df_f_4 = self.FA_Block4(df_f_4)  # 8, 96, 192, 192
        y1 = self.predict_layer_1(df_f_4)  # 8, 1, 384, 384
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(df_f_1), size=384, mode='bilinear')
        return y1, y2, y3, y4

    def load_pre(self, pre_model):
        msg = self.backbone.load_state_dict(torch.load(pre_model), strict=False)  # for swin: model, for eva02 :state_dict
        print(f"MAE loading pre_model ${pre_model}")
        print(msg)
        pass

    def frozen(self):
        self.frozen_exclude = ['all']
        if 'all' in self.frozen_exclude:
            print('All params will be updated')
            all_parameters = sum(p.numel() for _, p in self.backbone.named_parameters())
            print('number of all params (M): %.2f' % (all_parameters / 1.e6))
            print('updated params / all params: 100%')
            return
        else:
            updated_param = []
            for name, param in self.backbone.named_parameters():
                for i in self.frozen_exclude:
                    if i in name:
                        updated_param.append(name)
            assert len(updated_param) != 0
            print('The following parameters will be updated:{}'.format(updated_param))
            for name, param in self.backbone.named_parameters():
                if name in updated_param:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            n_parameters = sum(p.numel() for _, p in self.backbone.named_parameters() if p.requires_grad)
            all_parameters = sum(p.numel() for _, p in self.backbone.named_parameters())
            print('number of params (M) to be updated: %.2f' % (n_parameters / 1.e6))
            print('number of all params (M): %.2f' % (all_parameters / 1.e6))
            print('updated params / all params: %.2f %%' % ((n_parameters / 1.e6) / (all_parameters / 1.e6) * 100))

class EVA02CPNet(nn.Module):
    def __init__(self):
        super(EVA02CPNet, self).__init__()
        self.backbone = EVA2(
        # self.backbone = EVA2_ivt(
            embed_dim=1024,
            img_size=384,
            patch_size=16,
            in_chans=3,
            depth=24,
            num_heads=16,
            mlp_ratio=4 * 2 / 3,  # GLU default
            out_indices=[7, 11, 15, 23],
            qkv_bias=True,
            drop_path_rate=0.2,
            init_values=None,
            use_checkpoint=False,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            rope=True,
            pt_hw_seq_len=16,
            intp_freq=True,
            subln=True,
            xattn=True,
            naiveswiglu=True,
            pretrained="/data2/zym/Projects/mmdetection3d-main/pretrain/eva02_L_pt_m38m_p14to16_converted.pt",
            norm_layer=backbone_norm_cfg,
        )
        self.FA_Block2 = Block(dim=384)  # 384
        self.FA_Block3 = Block(dim=192)  # 192
        self.FA_Block4 = Block(dim=96)  # 96
        self.predtrans2 = nn.Conv2d(192, 1, kernel_size=3, padding=1)  # 192
        self.predtrans3 = nn.Conv2d(384, 1, kernel_size=3, padding=1)  # 384
        self.predtrans4 = nn.Conv2d(768, 1, kernel_size=3, padding=1)  # 768
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 = nn.Sequential(  # 1536
            nn.Conv2d(in_channels=1024, out_channels=768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(  # 1536
            nn.Conv2d(in_channels=1792, out_channels=384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_3 = nn.Sequential(  # 768
            nn.Conv2d(in_channels=1408, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_4 = nn.Sequential(  # 384
            nn.Conv2d(in_channels=1216, out_channels=96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            self.upsample2
        )
        self.predict_layer_1 = nn.Sequential(  # 96
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x, z):
        x = self.backbone(x, z)
        # [8, 1024, 96, 96] [8, 1024, 48, 48] [8, 1024, 24, 24] [8, 1024, 12, 12]
        # [192, 96, 96], [384, 48, 48], [768, 24, 24], [1536, 12, 12]
        xf_1 = x[3]  # 8 1024 12 12   # 1536,12,12
        xf_2 = x[2]  # 8 1024 24 24   # 768,24,24
        xf_3 = x[1]  # 8 1024 48 48   # 384,48,48
        xf_4 = x[0]  # 8 1024 96 96   # 192,96,96

        df_f_1 = self.deconv_layer_1(xf_1)  # 8, 768, 24, 24

        xc_1_2 = torch.cat((df_f_1, xf_2), 1)  # 8, 1792, 24, 24
        df_f_2 = self.deconv_layer_2(xc_1_2)  # 8, 384, 48, 48
        df_f_2 = self.FA_Block2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)  # 8, 1408, 48, 48
        df_f_3 = self.deconv_layer_3(xc_1_3)  # 8, 192, 96, 96
        df_f_3 = self.FA_Block3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)  # 8, 1216, 96, 96
        df_f_4 = self.deconv_layer_4(xc_1_4)  # 8, 96, 192, 192
        df_f_4 = self.FA_Block4(df_f_4)  # 8, 96, 192, 192
        y1 = self.predict_layer_1(df_f_4)  # 8, 1, 384, 384
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(df_f_1), size=384, mode='bilinear')
        return y1, y2, y3, y4

    def load_pre(self, pre_model):
        # self.backbone.load_state_dict(torch.load(pre_model), strict=False)  # for swin: model, for eva02 :state_dict
        # print(f"EVA02 loading pre_model ${pre_model}")
        pass


class ViTCPHead(nn.Module):
    def __init__(self):
        super(EVA02CPNet, self).__init__()
        self.backbone = SwinTransformer(embed_dim=192, depths=[2,2,18,2], num_heads=[6,12,24,48])
        # self.backbone = SwinTransformer_ivt(embed_dim=192, depths=[2,2,18,2], num_heads=[6,12,24,48])
        self.FA_Block2 = Block(dim=384)
        self.FA_Block3 = Block(dim=192)
        self.FA_Block4 = Block(dim=96)
        self.predtrans2 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(384, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(768, 1, kernel_size=3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=768, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(768),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=1536, out_channels=384, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=192, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            self.upsample2
        )
        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x, z):
        x = self.backbone(x, z)   # [192, 96, 96], [384, 48, 48], [768, 24, 24], [1536, 12, 12]
        xf_1 = x[3]  # 1536,12,12
        xf_2 = x[2]  # 768,24,24
        xf_3 = x[1]  # 384,48,48
        xf_4 = x[0]  # 192,96,96

        df_f_1 = self.deconv_layer_1(xf_1)

        xc_1_2 = torch.cat((df_f_1, xf_2), 1)
        df_f_2 = self.deconv_layer_2(xc_1_2)
        df_f_2 = self.FA_Block2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)
        df_f_3 = self.deconv_layer_3(xc_1_3)
        df_f_3 = self.FA_Block3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)
        df_f_4 = self.deconv_layer_4(xc_1_4)
        df_f_4 = self.FA_Block4(df_f_4)
        y1 = self.predict_layer_1(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(df_f_1), size=384, mode='bilinear')
        return y1, y2, y3, y4

    def load_pre(self, pre_model):
        self.backbone.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"SwinTransformer loading pre_model ${pre_model}")

# class CPNetL(nn.Module):  # useless
#     def __init__(self):
#         super(CPNetL, self).__init__()
#         # self.backbone = VisionTransformer(img_size=(512, 512),
#         #                                   embed_dims=1024,
#         #                                   num_layers=16,
#         #                                   num_heads=16,
#         #                                   out_indices=(9, 14, 19, 23),
#         #                                   output_cls_token=True,
#         #                                   frozen_exclude=['all'],  # fft
#         #                                   with_cls_token=True,
#         #                                   init_cfg=init_cfg)
#         self.backbone = SwinTransformer(embed_dim=192, depths=[2,2,18,2], num_heads=[6,12,24,48])
#         # self.backbone = SwinTransformer_ivt(embed_dim=192, depths=[2,2,18,2], num_heads=[6,12,24,48])
#         self.FA_Block2 = Block(dim=384)
#         self.FA_Block3 = Block(dim=192)
#         self.FA_Block4 = Block(dim=96)
#         self.predtrans2 = nn.Conv2d(192, 1, kernel_size=3, padding=1)
#         self.predtrans3 = nn.Conv2d(384, 1, kernel_size=3, padding=1)
#         self.predtrans4 = nn.Conv2d(768, 1, kernel_size=3, padding=1)
#         self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
#         self.deconv_layer_1 = nn.Sequential(
#             nn.Conv2d(in_channels=1536, out_channels=768, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(768),
#             nn.GELU(),
#             self.upsample2
#         )
#         self.deconv_layer_2 = nn.Sequential(
#             nn.Conv2d(in_channels=1536, out_channels=384, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(384),
#             nn.GELU(),
#             self.upsample2
#         )
#         self.deconv_layer_3 = nn.Sequential(
#             nn.Conv2d(in_channels=768, out_channels=192, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(192),
#             nn.GELU(),
#             self.upsample2
#         )
#         self.deconv_layer_4 = nn.Sequential(
#             nn.Conv2d(in_channels=384, out_channels=96, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(96),
#             nn.GELU(),
#             self.upsample2
#         )
#         self.predict_layer_1 = nn.Sequential(
#             nn.Conv2d(in_channels=96, out_channels=48, kernel_size=3, padding=1, bias=False),
#             nn.BatchNorm2d(48),
#             nn.GELU(),
#             self.upsample2,
#             nn.Conv2d(in_channels=48, out_channels=1, kernel_size=3, padding=1, bias=True),
#         )
#
#     def forward(self, x, z):
#         x = self.backbone(x, z)   # [192, 96, 96], [384, 48, 48], [768, 24, 24], [1536, 12, 12]
#         r1 = x[3]  # 1536,12,12
#         r2 = x[2]  # 768,24,24
#         r3 = x[1]  # 384,48,48
#         r4 = x[0]  # 192,96,96
#
#         r3_up = F.interpolate(self.dwc3(r3), size=96, mode='bilinear')
#         r2_up = F.interpolate(self.dwc2(r2), size=48, mode='bilinear')
#         r1_up = F.interpolate(self.dwc1(r1), size=24, mode='bilinear')
#
#         r1_con = torch.cat((r1, r1), 1)
#         r1_con = self.dwcon_1(r1_con)
#         r2_con = torch.cat((r2, r1_up), 1)
#         r2_con = self.dwcon_2(r2_con)
#         r3_con = torch.cat((r3, r2_up), 1)
#         r3_con = self.dwcon_3(r3_con)
#         r4_con = torch.cat((r4, r3_up), 1)
#         r4_con = self.dwcon_4(r4_con)
#
#
#         df_f_1 = self.deconv_layer_1(xf_1)
#
#         xc_1_2 = torch.cat((df_f_1, xf_2), 1)
#         df_f_2 = self.deconv_layer_2(xc_1_2)
#         df_f_2 = self.FA_Block2(df_f_2)
#
#         xc_1_3 = torch.cat((df_f_2, xf_3), 1)
#         df_f_3 = self.deconv_layer_3(xc_1_3)
#         df_f_3 = self.FA_Block3(df_f_3)
#
#         xc_1_4 = torch.cat((df_f_3, xf_4), 1)
#         df_f_4 = self.deconv_layer_4(xc_1_4)
#         df_f_4 = self.FA_Block4(df_f_4)
#         y1 = self.predict_layer_1(df_f_4)
#         y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
#         y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
#         y4 = F.interpolate(self.predtrans4(df_f_1), size=384, mode='bilinear')
#         return y1, y2, y3, y4
#
#     def load_pre(self, pre_model):
#         self.backbone.load_state_dict(torch.load(pre_model)['model'], strict=False)
#         print(f"SwinTransformer loading pre_model ${pre_model}")  #


class CPNet(nn.Module):
    def __init__(self):
        super(CPNet, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)

        self.CA_SA_Enhance_1 = CoordAtt(2048, 2048)
        self.CA_SA_Enhance_2 = CoordAtt(1024, 1024)
        self.CA_SA_Enhance_3 = CoordAtt(512, 512)
        self.CA_SA_Enhance_4 = CoordAtt(256, 256)

        self.FA_Block2 = Block(dim=256)
        self.FA_Block3 = Block(dim=128)
        self.FA_Block4 = Block(dim=64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.predtrans2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.dwc3 = conv3x3_bn_relu(256, 128)
        self.dwc2 = conv3x3_bn_relu(512, 256)
        self.dwc1 = conv3x3_bn_relu(1024, 512)
        self.dwcon_1 = conv3x3_bn_relu(2048, 1024)
        self.dwcon_2 = conv3x3_bn_relu(1024, 512)
        self.dwcon_3 = conv3x3_bn_relu(512, 256)
        self.dwcon_4 = conv3x3_bn_relu(256, 128)
        self.conv43 = conv3x3_bn_relu(128, 256, s=2)
        self.conv32 = conv3x3_bn_relu(256, 512, s=2)
        self.conv21 = conv3x3_bn_relu(512, 1024, s=2)



    def forward(self,x ,d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r4 = rgb_list[0]
        r3 = rgb_list[1]
        r2 = rgb_list[2]
        r1 = rgb_list[3]
        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]

        r3_up = F.interpolate(self.dwc3(r3), size=96, mode='bilinear')
        r2_up = F.interpolate(self.dwc2(r2), size=48, mode='bilinear')
        r1_up = F.interpolate(self.dwc1(r1), size=24, mode='bilinear')
        d3_up = F.interpolate(self.dwc3(d3), size=96, mode='bilinear')
        d2_up = F.interpolate(self.dwc2(d2), size=48, mode='bilinear')
        d1_up = F.interpolate(self.dwc1(d1), size=24, mode='bilinear')

        r1_con = torch.cat((r1, r1), 1)
        r1_con = self.dwcon_1(r1_con)
        d1_con = torch.cat((d1, d1), 1)
        d1_con = self.dwcon_1(d1_con)
        r2_con = torch.cat((r2, r1_up), 1)
        r2_con = self.dwcon_2(r2_con)
        d2_con = torch.cat((d2, d1_up), 1)
        d2_con = self.dwcon_2(d2_con)
        r3_con = torch.cat((r3, r2_up), 1)
        r3_con = self.dwcon_3(r3_con)
        d3_con = torch.cat((d3, d2_up), 1)
        d3_con = self.dwcon_3(d3_con)
        r4_con = torch.cat((r4, r3_up), 1)
        r4_con = self.dwcon_4(r4_con)
        d4_con = torch.cat((d4, d3_up), 1)
        d4_con = self.dwcon_4(d4_con)


        xf_1 = self.CA_SA_Enhance_1(r1_con, d1_con)  # 1024,12,12
        xf_2 = self.CA_SA_Enhance_2(r2_con, d2_con)  # 512,24,24
        xf_3 = self.CA_SA_Enhance_3(r3_con, d3_con)  # 256,48,48
        xf_4 = self.CA_SA_Enhance_4(r4_con, d4_con)  # 128,96,96


        df_f_1 = self.deconv_layer_1(xf_1)  # 512, 24, 24

        xc_1_2 = torch.cat((df_f_1, xf_2), 1)  # 1024, 24, 24
        df_f_2 = self.deconv_layer_2(xc_1_2)
        df_f_2 = self.FA_Block2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)
        df_f_3 = self.deconv_layer_3(xc_1_3)
        df_f_3 = self.FA_Block3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)
        df_f_4 = self.deconv_layer_4(xc_1_4)
        df_f_4 = self.FA_Block4(df_f_4)
        y1 = self.predict_layer_1(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(df_f_1), size=384, mode='bilinear')
        return y1,y2,y3,y4

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_end = nn.Conv2d(oup, oup // 2, kernel_size=1, stride=1, padding=0)
        self.self_SA_Enhance = SA_Enhance()

    def forward(self, rgb, depth):
        x = torch.cat((rgb, depth), dim=1)

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out_ca = x * a_w * a_h
        out_sa = self.self_SA_Enhance(out_ca)
        out = x.mul(out_sa)
        out = self.conv_end(out)

        return out

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x
