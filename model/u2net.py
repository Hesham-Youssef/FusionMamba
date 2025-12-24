import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba
from mamba_ssm.modules.mamba_simple import Mamba
import math


# =============================================================================
# FIXED VERSION - Addresses checkerboard, statistics, and convergence issues
# =============================================================================


class DynamicRangeNorm(nn.Module):
    """Lightweight normalization for pre-normalized inputs"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
    def forward(self, x):
        return self.gamma * x + self.beta


class ExposureAwareAttention(nn.Module):
    """Conservative attention to prevent range violations"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels * 2, channels // reduction, 1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.fc(combined))
        
        # FIX: More conservative scaling [0.85, 1.15]
        return x * (attention * 0.3 + 0.85)


class Up(nn.Module):
    """FIXED: No more checkerboard artifacts!"""
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        
        # FIX: Use Upsample + Conv instead of ConvTranspose2d
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),  # Regular conv after upsample
            DynamicRangeNorm(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1),
            DynamicRangeNorm(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            DynamicRangeNorm(out_channels)
        )
        
        self.attention = ExposureAwareAttention(out_channels)
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.attention(x)
        return x + x1 * 0.5  # Dampen residual


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, scale, scale, 0),
            DynamicRangeNorm(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            DynamicRangeNorm(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.down(x)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, scale=2, sample_mode='down'):
        super().__init__()
        self.fm = FusionMamba(in_channels, H, W)
        
        if sample_mode == 'down':
            self.sample = Down(in_channels, out_channels, scale)
            self.sum_sample = Down(in_channels, out_channels, scale)
        elif sample_mode == 'up':
            self.sample = Up(in_channels, out_channels, scale)
            self.sum_sample = Up(in_channels, out_channels, scale)

    def forward(self, img1, img2, img1_sum, img2_sum, img1_pre=None, img2_pre=None):
        img1, img2 = self.fm(img1, img2, img1_sum, img2_sum)

        if img1_pre is None:
            img1_skip = img1
            img2_skip = img2
            img1 = self.sample(img1)
            img2 = self.sample(img2)
            img1_sum = self.sum_sample(img1_sum)
            img2_sum = self.sum_sample(img2_sum)
            return img1, img2, img1_sum, img2_sum, img1_skip, img2_skip
        else:
            img1 = self.sample(img1, img1_pre)
            img2 = self.sample(img2, img2_pre)
            img1_sum = self.sum_sample(img1_sum, img1_pre)
            img2_sum = self.sum_sample(img2_sum, img2_pre)
            return img1, img2, img1_sum, img2_sum


class HDRHead(nn.Module):
    """FIXED: Constrained output for [-1, 1] range"""
    def __init__(self, dim, out_channels):
        super().__init__()
        num_groups = math.gcd(8, dim)  # largest divisor ≤ 8
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GroupNorm(num_groups, dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            nn.GroupNorm(num_groups//2, dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, out_channels, 3, 1, 1)
        )
        
        # FIX: Conservative initialization
        nn.init.xavier_uniform_(self.conv[-1].weight, gain=0.5)
        nn.init.constant_(self.conv[-1].bias, 0.0)
        
        # FIX: Soft tanh constraint to keep output near [-1, 1]
        self.output_scale = nn.Parameter(torch.tensor(1.2))
        
    def forward(self, x):
        output = self.conv(x)
        # Soft constraint: allows slight excursion beyond [-1, 1] but pulls back
        return torch.tanh(output) * self.output_scale


class SpeAttention(nn.Module):
    """Conservative spectral attention"""
    def __init__(self, channels=32):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        self.block = nn.Sequential(
            nn.Linear(2, channels),
            nn.LayerNorm(channels),
            Mamba(channels, expand=1, d_state=8, bimamba_type='v2', 
                  if_devide_out=True, use_norm=True),
            nn.Linear(channels, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input = input.contiguous()
        avg_out = self.avg_pool(input).squeeze(-1)
        max_out = self.max_pool(input).squeeze(-1)
        combined = torch.cat([avg_out, max_out], dim=-1).contiguous()
        
        attention = self.block(combined).unsqueeze(-1)
        
        # FIX: Very conservative range [0.95, 1.05]
        attention = attention * 0.1 + 0.95
        
        return attention


class U2Net(nn.Module):

    def __init__(self, dim, img1_dim, img2_dim, H=64, W=64):
        super().__init__()

        # Input projection layers
        self.raise_img1_dim = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            DynamicRangeNorm(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.raise_img2_dim = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            DynamicRangeNorm(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.raise_img1_sum_dim = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            DynamicRangeNorm(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.raise_img2_sum_dim = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            DynamicRangeNorm(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.to_hdr = HDRHead(dim, img2_dim // 2)

        # Dimensions for each stage
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        # Main U-Net body
        self.stage0 = Stage(dim0, dim1, H, W, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, H//2, W//2, sample_mode='down')
        self.stage2 = Stage(dim2, dim1, H//4, W//4, sample_mode='up')
        self.stage3 = Stage(dim1, dim0, H//2, W//2, sample_mode='up')
        self.stage4 = FusionMamba(dim0, H, W, depth=3, final=True)
        
        # FIX: Simpler, lighter feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            DynamicRangeNorm(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # FIX: Use spectral attention for color
        self.spe_attn1 = SpeAttention(channels=dim)
        self.spe_attn2 = SpeAttention(channels=dim)
        
        # FIX: Very small color blend
        self.color_blend_alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, img1, img2, sum1, sum2):
        img1 = self.raise_img1_dim(img1)
        img2 = self.raise_img2_dim(img2)
        sum1 = self.raise_img1_sum_dim(sum1)
        sum2 = self.raise_img2_sum_dim(sum2)

        # Store originals
        img1_orig = img1
        img2_orig = img2

        # U-Net encoder
        img1, img2, sum1, sum2, img1_skip0, img2_skip0 = self.stage0(img1, img2, sum1, sum2)
        img1, img2, sum1, sum2, img1_skip1, img2_skip1 = self.stage1(img1, img2, sum1, sum2)
        
        # U-Net decoder
        img1, img2, sum1, sum2 = self.stage2(img1, img2, sum1, sum2, img1_skip1, img2_skip1)
        img1, img2, sum1, sum2 = self.stage3(img1, img2, sum1, sum2, img1_skip0, img2_skip0)
        
        # Final fusion
        output = self.stage4(img1, img2, sum1, sum2)

        # FIX: Single refinement pass
        output = output + self.feature_refine(output) * 0.2

        # FIX: Minimal color preservation
        attn1 = self.spe_attn1(img1_orig)
        attn2 = self.spe_attn2(img2_orig)
        color_residual = (img1_orig * attn1 + img2_orig * attn2) * 0.5
        output = output + color_residual * self.color_blend_alpha

        # Generate HDR output (now constrained by tanh)
        output = self.to_hdr(output)

        output = torch.clamp(output, -1.0, 1.0)

        return output