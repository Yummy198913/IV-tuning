import torch.nn.functional as F
import torch.nn as nn
import torch

class SimAM(torch.nn.Module):
    def __init__(self, channels=None, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):

        # b, c, h, w = x.size()
        b, c, w, h = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )
        return x * self.activaton(y)


class HybridOperation(nn.Module):
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


class HybridAdapter(nn.Module):
    def __init__(self,
                 in_dim=768,
                 hidden_dim=64):
        super().__init__()

        self.project1 = nn.Linear(in_dim, hidden_dim)
        self.nonlinear = F.gelu
        self.project2 = nn.Linear(hidden_dim, in_dim)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = HybridOperation(hidden_dim)
        self.sft = SimpleFeatureTransform(in_dim=in_dim, norm_config=True)

    def forward(self, x, hw_shapes=None):
        identity = x
        x = self.sft(x)
        project1 = self.project1(x)

        b, n, c = project1.shape
        h, w = hw_shapes
        project1 = project1.reshape(b, h, w, c).permute(0, 3, 1, 2)
        project1 = self.adapter_conv(project1)
        project1 = project1.permute(0, 2, 3, 1).reshape(b, n, c)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2


class SimpleFeatureTransform(nn.Module):
    def __init__(self, in_dim, norm_config=True):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(in_dim))
        self.beta = nn.Parameter(torch.zeros(in_dim))
        self.norm_config = norm_config
        if self.norm_config:
            self.norm = nn.LayerNorm(in_dim)

    def forward(self, x):
        if self.norm_config:
            return self.norm(x) * self.gamma + self.beta
        else:
            return x * self.gamma + self.beta


# fixme
class InternalPromptAdapter(nn.Module):
    def __init__(self, in_dim=768, hide_dim=8, ):
        super().__init__()
        self.down = nn.Linear(in_dim, hide_dim)
        self.up = nn.Linear(hide_dim, in_dim)
        self.act = nn.GELU()
        self.act_prompt = nn.GELU()
        self.simam = SimAM()
        self.sft1 = SimpleFeatureTransform(in_dim=in_dim, norm_config=True)
        self.sft2 = SimpleFeatureTransform(in_dim=in_dim, norm_config=False)

    def forward(self, x, hw_shapes):
        x = self.down(self.sft1(x))
        b, n, c = x.shape
        h, w = hw_shapes
        x = x.reshape(b, w, h, c).permute(0, 3, 1, 2)  # bcwh
        x = self.simam(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1).reshape(b, n, c)
        x = self.sft2(self.up(x))
        return self.act_prompt(x)


class EMPG(nn.Module):
    def __init__(self, inplanes=None, hide_channel=8):
        super(EMPG, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)
        self.simam = SimAM(hide_channel)
        self.HO = HybridOperation(hide_channel)

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
        x0 = self.HO(x0) + self.simam(x1)

        return self.conv1x1(x0)


def token2feature(tokens, H, W):
    B, L, D = tokens.shape
    x = tokens.permute(0, 2, 1).view(B, D, H, W).contiguous()
    return x

def feature2token(x):
    B, C, H, W = x.shape
    L = H * W
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens