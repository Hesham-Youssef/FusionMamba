import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba
from mamba_ssm.modules.mamba_simple import Mamba
import math

class PixelShuffle(nn.Module):
    def __init__(self, dim, scale):
        super().__init__()
        self.upsamle = nn.Sequential(
            nn.Conv2d(dim, dim*(scale**2), 3, 1, 1, bias=False),
            nn.PixelShuffle(scale)
        )

    def forward(self, x):
        return self.upsamle(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, scale, scale, 0),
            nn.LeakyReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
            nn.LeakyReLU()
            )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = x1 + x2
        return self.conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, scale, scale, 0),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.down(x)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, scale=2, sample_mode='down'):
        super().__init__()
        self.fm = FusionMamba(in_channels, H, W)
        
        # use different paths for the summaries
        if sample_mode == 'down':
            self.sample = Down(in_channels, out_channels, scale)
            self.sum_sample = Down(in_channels, out_channels, scale)
        elif sample_mode == 'up':
            self.sample = Up(in_channels, out_channels, scale)
            self.sum_sample = Up(in_channels, out_channels, scale)

    def forward(self, img1, img2, img1_sum, img2_sum, img1_pre=None, img2_pre=None,):
        img1, img2 = self.fm(img1, img2, img1_sum, img2_sum)
        if img1_pre is None:
            # inside down
            img1_skip = img1
            img2_skip = img2
            img1 = self.sample(img1)
            img2 = self.sample(img2)
            
            img1_sum = self.sum_sample(img1_sum)
            img2_sum = self.sum_sample(img2_sum)
            return img1, img2, img1_sum, img2_sum, img1_skip, img2_skip
        else:
            #inside up
            img1 = self.sample(img1, img1_pre)
            img2 = self.sample(img2, img2_pre)
            
            img1_sum = self.sum_sample(img1_sum, img1_pre)
            img2_sum = self.sum_sample(img2_sum, img2_pre)
            return img1, img2, img1_sum, img2_sum


class SpeAttention(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))

        self.block = nn.Sequential(
            nn.Linear(1, channels),
            nn.LayerNorm(channels),
            Mamba(channels, expand=1, d_state=8, bimamba_type='v2', if_devide_out=True, use_norm=True),
            nn.Linear(channels, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        input = self.pooling(input)
        output = self.block(input.squeeze(-1)).unsqueeze(-1)
        return self.sigmoid(output)


class U2Net(nn.Module):
    def __init__(self, dim, img1_dim, img2_dim, H=64, W=64):
        super().__init__()

        # ---------- Input lifting ----------
        self.raise_img1 = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.raise_img2 = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.raise_sum1 = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        self.raise_sum2 = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )

        # ---------- Main backbone ----------
        dim0, dim1, dim2 = dim, dim * 2, dim * 4
        self.stage0 = Stage(dim0, dim1, H, W, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, H // 2, W // 2, sample_mode='down')
        self.stage2 = Stage(dim2, dim1, H // 4, W // 4, sample_mode='up')
        self.stage3 = Stage(dim1, dim0, H // 2, W // 2, sample_mode='up')
        self.stage4 = FusionMamba(dim0, H, W, final=True)

        # ---------- Output head ----------
        self.to_hr = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, img2_dim, 3, 1, 1)
        )

        # ---------- Safe scale predictor ----------
        self.scale_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(img1_dim + img2_dim, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, img2_dim, 1)
        )
        
        with torch.no_grad():
            self.scale_net[-1].weight *= 0.01
            self.scale_net[-1].bias.fill_(math.log(50.0))

        # ---------- Skip (residual, not competing) ----------
        self.skip_conv = nn.Sequential(
            nn.Conv2d(img1_dim, img2_dim, 1),
            nn.LeakyReLU(inplace=True)
        )

        self.skip_scale_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(img1_dim, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1)
        )

        with torch.no_grad():
            self.skip_scale_net[-1].weight *= 0.01
            self.skip_scale_net[-1].bias.fill_(math.log(30.0))

    def forward(self, img1, img2, sum1, sum2):
        org1, org2 = img1, img2

        # ---------- Adaptive scale ----------
        scale_input = torch.cat([org1, org2], dim=1)
        log_scale = self.scale_net(scale_input)
        adaptive_scale = torch.exp(log_scale).clamp(1e-3, 300.0)

        skip_log_scale = self.skip_scale_net(org1)
        skip_scale = torch.exp(skip_log_scale).clamp(1e-3, 300.0)

        # ---------- Backbone ----------
        img1 = self.raise_img1(img1)
        img2 = self.raise_img2(img2)
        sum1 = self.raise_sum1(sum1)
        sum2 = self.raise_sum2(sum2)

        img1, img2, sum1, sum2, s0_1, s0_2 = self.stage0(img1, img2, sum1, sum2)
        img1, img2, sum1, sum2, s1_1, s1_2 = self.stage1(img1, img2, sum1, sum2)
        img1, img2, sum1, sum2 = self.stage2(img1, img2, sum1, sum2, s1_1, s1_2)
        img1, img2, sum1, sum2 = self.stage3(img1, img2, sum1, sum2, s0_1, s0_2)

        feat = self.stage4(img1, img2, sum1, sum2)
        pred = self.to_hr(feat)

        # ---------- Linear HDR ----------
        pred = F.softplus(pred) * adaptive_scale

        # ---------- Residual skip ----------
        skip = F.softplus(self.skip_conv(org1)) * skip_scale
        output_linear = pred + skip

        output_log = torch.log1p(output_linear)

        return output_log, output_linear, adaptive_scale, skip_scale