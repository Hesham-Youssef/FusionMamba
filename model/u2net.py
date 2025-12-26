import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba
from mamba_ssm.modules.mamba_simple import Mamba


class SimpleAttention(nn.Module):
    """Lightweight channel attention without over-compression"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        attention = self.fc(self.avg_pool(x))
        return x * attention * 0.5 + x * 0.5  # Gentle modulation


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GELU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1),
            nn.GELU()
        )
        
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, scale, scale, 0),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GELU()
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
        img1, img2, img1_sum, img2_sum = self.fm(img1, img2, img1_sum, img2_sum)

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


class U2Net(nn.Module):
    """Simplified U2Net focused on preserving HDR dynamic range"""

    def __init__(self, dim, img1_dim, img2_dim, H=64, W=64, debug=False):
        super().__init__()
        self.debug = debug
        self.register_buffer('step', torch.tensor(0))

        # Input projection - simple and direct
        self.raise_img1_dim = nn.Conv2d(img1_dim, dim, 3, 1, 1)
        self.raise_img2_dim = nn.Conv2d(img2_dim, dim, 3, 1, 1)
        self.raise_img1_sum_dim = nn.Conv2d(img1_dim, dim, 3, 1, 1)
        self.raise_img2_sum_dim = nn.Conv2d(img2_dim, dim, 3, 1, 1)

        # Dimensions for each stage
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        # Main U-Net body
        self.stage0 = Stage(dim0, dim1, H, W, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, H//2, W//2, sample_mode='down')
        self.stage2 = Stage(dim2, dim1, H//4, W//4, sample_mode='up')
        self.stage3 = Stage(dim1, dim0, H//2, W//2, sample_mode='up')
        self.stage4 = FusionMamba(dim0, H, W, depth=1, final=True)
        
        # Simple refinement
        self.refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        
        # Attention for exposure blending
        self.attention = SimpleAttention(dim)
        
        # Final output projection - CRITICAL: preserve HDR range
        self.final_proj = nn.Sequential(
            nn.Conv2d(dim, img2_dim, 3, 1, 1),
            nn.Tanh()
        )
        
        # Initialize final_proj to preserve signal
        with torch.no_grad():
            # Initialize to approximate identity mapping
            nn.init.xavier_uniform_(self.final_proj[0].weight, gain=0.5)
            if self.final_proj[0].bias is not None:
                nn.init.zeros_(self.final_proj[0].bias)

    def _print_range(self, name, tensor):
        """Debug utility to monitor value ranges"""
        # if self.debug and self.step % 100 == 0:
        #     print(f"{name:25s}: [{tensor.min():7.3f}, {tensor.max():7.3f}] "
        #           f"μ={tensor.mean():7.3f} σ={tensor.std():7.3f}")

    def forward(self, img1, img2, sum1, sum2):
        if self.training:
            self.step += 1
        
        # Project to feature space
        img1 = self.raise_img1_dim(img1)
        img2 = self.raise_img2_dim(img2)
        sum1 = self.raise_img1_sum_dim(sum1)
        sum2 = self.raise_img2_sum_dim(sum2)

        img1_orig = img1
        img2_orig = img2
        
        self._print_range("Input img1", img1_orig)
        self._print_range("Input img2", img2_orig)

        # U-Net encoder
        img1, img2, sum1, sum2, img1_skip0, img2_skip0 = self.stage0(img1, img2, sum1, sum2)
        img1, img2, sum1, sum2, img1_skip1, img2_skip1 = self.stage1(img1, img2, sum1, sum2)
        
        # U-Net decoder
        img1, img2, sum1, sum2 = self.stage2(img1, img2, sum1, sum2, img1_skip1, img2_skip1)
        img1, img2, sum1, sum2 = self.stage3(img1, img2, sum1, sum2, img1_skip0, img2_skip0)
        
        # Final fusion
        output = self.stage4(img1, img2, sum1, sum2)
        self._print_range("After stage4", output)
        
        # Add skip connection from input
        output = output + img1_orig * 0.3
        self._print_range("After skip", output)
        
        # Refinement
        refined = self.refine(output)
        output = output + refined * 0.5
        self._print_range("After refine", output)
        
        # Gentle attention modulation
        output = self.attention(output)
        self._print_range("After attention", output)
        
        # Final projection to output space
        output = self.final_proj(output)
        self._print_range("After final_proj", output)
        
        # NO CLAMP - let HDR values be HDR!
        # The loss function (tonemapping) will handle the range
        # output = nn.Tanh(output)
        return output