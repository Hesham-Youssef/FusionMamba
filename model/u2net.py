import torch
import torch.nn as nn
from torchinfo import summary
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba
from mamba_ssm.modules.mamba_simple import Mamba
import math


# =============================================================================
# OPTIMIZED FOR PRE-NORMALIZED INPUTS [-1, 1]
# =============================================================================
# Key changes for [-1, 1] normalized data:
# 1. DynamicRangeNorm → Simple affine transform (no mean/var normalization)
# 2. HDRHead → Tanh for [-1, 1] output (not Sigmoid!)
# 3. Zero-centered initialization and processing
# 4. Conservative attention to prevent range violations
# =============================================================================


class DynamicRangeNorm(nn.Module):
    """Lightweight normalization for pre-normalized inputs"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # For pre-normalized inputs, just use learnable scaling
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
    def forward(self, x):
        # Skip heavy normalization, just apply affine transform
        return self.gamma * x + self.beta


class ExposureAwareAttention(nn.Module):
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
        # FIX: Moderate attention for [-1, 1] range
        return x * (attention * 0.4 + 0.8)  # Range [0.8, 1.2]


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, scale, scale, 0),
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
        return x + x1


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
    """HDR head optimized for normalized inputs/outputs [-1, 1]"""
    def __init__(self, dim, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.InstanceNorm2d(dim, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            nn.InstanceNorm2d(dim // 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, out_channels, 3, 1, 1),
            nn.Tanh()  # Output [-1, 1] for normalized data
        )
        
        # Initialize final layer to output near 0 (center of range)
        nn.init.xavier_uniform_(self.conv[-2].weight, gain=0.3)
        nn.init.constant_(self.conv[-2].bias, 0.0)
        
    def forward(self, x):
        return self.conv(x)


class SpeAttention(nn.Module):
    """FIXED: Less aggressive attention scaling"""
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
        
        # FIX: Centered around 1.0 for [-1, 1] data
        attention = attention * 0.25 + 0.875  # Range [0.875, 1.125]
        
        return attention


class U2Net(nn.Module):

    def __init__(self, dim, img1_dim, img2_dim, H=64, W=64):
        super().__init__()

        # Input projection layers - keep some normalization for stability
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
        
        # Feature refinement - keep it light
        self.feature_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1),
                DynamicRangeNorm(dim),
                nn.LeakyReLU(0.2, inplace=True)
            ) for _ in range(2)
        ])
        
        self.final_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )
        
        # FIX: Use spectral attention for intelligent color preservation
        self.spe_attn1 = SpeAttention(channels=dim)
        self.spe_attn2 = SpeAttention(channels=dim)
        
        # For [-1, 1] normalized inputs, use small residual weight
        self.color_blend_alpha = nn.Parameter(torch.tensor(0.12))

    def forward(self, img1, img2, sum1, sum2):

        img1 = self.raise_img1_dim(img1)
        img2 = self.raise_img2_dim(img2)
        sum1 = self.raise_img1_sum_dim(sum1)
        sum2 = self.raise_img2_sum_dim(sum2)

        # Store original for residual
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

        # FIX: Lighter feature refinement with residual scaling
        for i, refine_layer in enumerate(self.feature_refine):
            scale = 0.3 ** (i + 1)  # Lighter scaling for normalized data
            output = output + refine_layer(output) * scale
        
        output = output + self.final_refine(output) * 0.15  # Slightly higher for [-1,1]

        # FIX: Conservative color preservation for normalized inputs
        # Add small amount of original color back
        attn1 = self.spe_attn1(img1_orig)
        attn2 = self.spe_attn2(img2_orig)
        
        # Weighted average of original inputs
        color_residual = (img1_orig * attn1 + img2_orig * attn2) * 0.5
        
        # Add small residual - network does most of the work
        output = output + color_residual * self.color_blend_alpha

        output_hdr = self.to_hdr(output)

        return output_hdr
    
    def print_statistics(self, img1, img2, sum1, sum2, output):
        """Debug helper - call this during evaluation to see what's happening"""
        print(f"\nInput ranges:")
        print(f"  img1: [{img1.min():.3f}, {img1.max():.3f}], mean: {img1.mean():.3f}")
        print(f"  img2: [{img2.min():.3f}, {img2.max():.3f}], mean: {img2.mean():.3f}")
        print(f"\nOutput ranges:")
        print(f"  output: [{output.min():.3f}, {output.max():.3f}], mean: {output.mean():.3f}")
        print(f"  color_blend_alpha: {self.color_blend_alpha.item():.3f}")
        
        # Check if output is clipped at boundaries
        near_min = (output < -0.95).float().mean() * 100
        near_max = (output > 0.95).float().mean() * 100
        print(f"  % pixels near -1.0: {near_min:.2f}%")
        print(f"  % pixels near +1.0: {near_max:.2f}%")
        if near_min > 5 or near_max > 5:
            print("  WARNING: Many pixels clipped at boundaries!")


# =============================================================================
# TUNING GUIDE FOR NORMALIZED INPUTS [-1, 1]
# =============================================================================
# 
# Your inputs are pre-normalized, so the network should do most of the work.
# The color_blend_alpha controls how much original color to inject.
#
# If output is GREY/DESATURATED:
# 1. Increase color_blend_alpha: 0.1 → 0.15 → 0.20 → 0.25
# 2. Increase SpeAttention range: attention * 0.3 + 0.85 → attention * 0.5 + 0.75
# 3. Reduce final_refine scaling: 0.1 → 0.05 (less smoothing)
#
# If output is OVERSATURATED/TOO BRIGHT:
# 1. Decrease color_blend_alpha: 0.1 → 0.05 → 0.03
# 2. Decrease SpeAttention range: attention * 0.3 + 0.85 → attention * 0.2 + 0.9
# 3. Check HDRHead initialization gain (currently 0.2, can go to 0.1)
# 4. Ensure your loss function isn't pushing outputs to extremes
#
# If output is CORRECT color but WRONG range:
# - HDRHead outputs [0, 1] via Sigmoid
# - If your targets are in different range, remove Sigmoid and add appropriate scaling
# - Example for [0, 10]: return self.conv(x) * 10
#
# Debug tips:
# - Print ranges: print(f"Output: {output.min():.3f} to {output.max():.3f}")
# - Visualize attention: save attn1.mean() and attn2.mean() to see if they're reasonable
# - Check if problem is in network vs HDRHead: print intermediate 'output' before to_hdr()
# =============================================================================