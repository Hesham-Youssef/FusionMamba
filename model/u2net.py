import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba
from mamba_ssm.modules.mamba_simple import Mamba


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
    def __init__(self, dim, img1_dim, img2_dim, H=64, W=64, scale=4):
        super().__init__()

        # self.upsample = PixelShuffle(img2_dim, scale)
        self.raise_img1_dim = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_img2_dim = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        
        self.raise_img1_sum_dim = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        self.raise_img2_sum_dim = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            nn.LeakyReLU()
        )
        
        
        self.to_hrimg2 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(dim, img2_dim, 3, 1, 1)
        )

        # dimension for each stage
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        # main body
        self.stage0 = Stage(dim0, dim1, H, W, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, H//2, W//2, sample_mode='down')
        
        self.stage2 = Stage(dim2, dim1, H//4, W//4, sample_mode='up')
        self.stage3 = Stage(dim1, dim0, H//2, W//2, sample_mode='up')
        self.stage4 = FusionMamba(dim0, H, W, final=True)

        self.img1_spe_attn = SpeAttention(dim)
        self.img2_spe_attn = SpeAttention(dim)
        
        
        skip_in_channels = img1_dim + img2_dim
        skip_mid = max(1, dim // 2)

        self.skip_gate = nn.Sequential(
            nn.Conv2d(skip_in_channels, dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, skip_mid, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_mid, 1, 1),
            nn.Sigmoid()
        )
        self.skip_alpha_param = nn.Parameter(torch.tensor(0.0))
        self.output_scale = nn.Parameter(torch.tensor(2.0))


    def forward(self, img1, img2, sum1, sum2):
        org_img1 = img1
        org_img2 = img2
        # img2 = self.upsample(img2)
        img1 = self.raise_img1_dim(img1)
        img2 = self.raise_img2_dim(img2)
        
        sum1 = self.raise_img1_sum_dim(sum1)
        sum2 = self.raise_img2_sum_dim(sum2)

        # main body
        img1, img2, sum1, sum2, img1_skip0, img2_skip0 = self.stage0(img1, img2, sum1, sum2)
        img1, img2, sum1, sum2, img1_skip1, img2_skip1 = self.stage1(img1, img2, sum1, sum2)
        
        img1, img2, sum1, sum2 = self.stage2(img1, img2, sum1, sum2, img1_skip1, img2_skip1)
        img1, img2, sum1, sum2 = self.stage3(img1, img2, sum1, sum2, img1_skip0, img2_skip0)
        output = self.stage4(img1, img2, sum1, sum2)

        img1_spe_attn = self.img1_spe_attn(org_img1)
        img2_spe_attn = self.img2_spe_attn(org_img2)

        # decoder → RGB
        output = self.to_hrimg2(output)

        # spectral attention
        output = output * (img1_spe_attn + img2_spe_attn)
        
        # output = output * self.output_scale.view(1, 1, 1, 1)

        gate_in = torch.cat([org_img1, output], dim=1)   # (B, 6, H, W)
        gate = self.skip_gate(gate_in)                    # (B, 1, H, W)

        skip_alpha = torch.sigmoid(self.skip_alpha_param)
        
        output = output + skip_alpha * gate * (org_img1 - output)

        return output