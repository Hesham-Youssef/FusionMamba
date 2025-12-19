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


class DynamicRangeNorm(nn.Module):
    """Normalizes features while preserving HDR characteristics"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
    def forward(self, x):
        # Preserve high dynamic range by normalizing per-channel
        mean = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class ExposureAwareAttention(nn.Module):
    """Attention mechanism that considers exposure levels"""
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
        # FIX: Ensure input is contiguous
        x = x.contiguous()
        
        avg_out = self.avg_pool(x)
        max_out = self.max_pool(x)
        
        # Combine average and max pooling for exposure awareness
        combined = torch.cat([avg_out, max_out], dim=1).contiguous()
        attention = self.sigmoid(self.fc(combined))
        
        return x * attention


class Up(nn.Module):
    """Improved upsampling with residual connections - FIXED for memory alignment"""
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
        # FIX: Ensure inputs are contiguous
        x1 = x1.contiguous()
        x2 = x2.contiguous()
        
        x1 = self.up(x1).contiguous()
        
        # FIX: Concatenate and ensure result is contiguous before conv
        x = torch.cat([x1, x2], dim=1).contiguous()
        x = self.conv(x).contiguous()
        x = self.attention(x).contiguous()
        
        return (x + x1).contiguous()  # Residual connection


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
        # FIX: Ensure input is contiguous
        x = x.contiguous()
        return self.down(x).contiguous()


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
        # FIX: Ensure all inputs are contiguous
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        img1_sum = img1_sum.contiguous()
        img2_sum = img2_sum.contiguous()
        
        img1, img2 = self.fm(img1, img2, img1_sum, img2_sum)
        
        # FIX: Ensure outputs from FusionMamba are contiguous
        img1 = img1.contiguous()
        img2 = img2.contiguous()

        if img1_pre is None:
            # Downsampling path
            img1_skip = img1.contiguous()
            img2_skip = img2.contiguous()
            img1 = self.sample(img1).contiguous()
            img2 = self.sample(img2).contiguous()
            
            img1_sum = self.sum_sample(img1_sum).contiguous()
            img2_sum = self.sum_sample(img2_sum).contiguous()
            return img1, img2, img1_sum, img2_sum, img1_skip, img2_skip
        else:
            # Upsampling path
            # FIX: Ensure skip connections are contiguous
            img1_pre = img1_pre.contiguous()
            img2_pre = img2_pre.contiguous()
            
            img1 = self.sample(img1, img1_pre).contiguous()
            img2 = self.sample(img2, img2_pre).contiguous()
            
            img1_sum = self.sum_sample(img1_sum, img1_pre).contiguous()
            img2_sum = self.sum_sample(img2_sum, img2_pre).contiguous()
            return img1, img2, img1_sum, img2_sum


class HDRHead(nn.Module):
    """Output head with proper HDR range handling"""
    def __init__(self, dim, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            DynamicRangeNorm(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            DynamicRangeNorm(dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim // 2, out_channels, 3, 1, 1)
        )
        
    def forward(self, x):
        # FIX: Ensure input is contiguous
        x = x.contiguous()
        
        # Output in log space, ensuring positive values
        out = self.conv(x)
        # Use softplus to ensure positive output (always >= 0)
        return F.softplus(out) + 1e-8


class SpeAttention(nn.Module):
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
            nn.Tanh()  # Changed from Sigmoid to allow negative adjustments
        )

    def forward(self, input):
        # FIX: Ensure input is contiguous
        input = input.contiguous()
        
        avg_out = self.avg_pool(input).squeeze(-1)
        max_out = self.max_pool(input).squeeze(-1)
        
        # Combine statistics
        combined = torch.cat([avg_out, max_out], dim=-1).contiguous()
        output = self.block(combined).unsqueeze(-1)
        
        # Scale from [-1, 1] to [0.5, 1.5] for stable multiplication
        output = output * 0.5 + 1.0
        return output
    

class U2Net(nn.Module):
    def __init__(self, dim, img1_dim, img2_dim, H=64, W=64, scale=4):
        super().__init__()

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
        
        self.to_hrimg2 = HDRHead(dim, img2_dim)

        # dimension for each stage
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        # main body
        self.stage0 = Stage(dim0, dim1, H, W, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, H//2, W//2, sample_mode='down')
        
        self.stage2 = Stage(dim2, dim1, H//4, W//4, sample_mode='up')
        self.stage3 = Stage(dim1, dim0, H//2, W//2, sample_mode='up')
        self.stage4 = FusionMamba(dim0, H, W, depth=3, final=True)

        self.img1_spe_attn = SpeAttention(dim)
        self.img2_spe_attn = SpeAttention(dim)
        
        # Additional feature refinement
        self.feature_refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim, dim, 3, 1, 1),
                DynamicRangeNorm(dim),
                nn.LeakyReLU(0.2, inplace=True)
            ) for _ in range(3)
        ])
        
        # Final refinement
        self.final_refine = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            DynamicRangeNorm(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(dim, dim, 1, 1, 0)
        )

    def forward(self, img1, img2, sum1, sum2):
        # FIX: Ensure all inputs are contiguous at the start
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        sum1 = sum1.contiguous()
        sum2 = sum2.contiguous()
        
        org_img1 = img1.contiguous()
        org_img2 = img2.contiguous()

        img1 = self.raise_img1_dim(img1).contiguous()
        img2 = self.raise_img2_dim(img2).contiguous()
        
        sum1 = self.raise_img1_sum_dim(sum1).contiguous()
        sum2 = self.raise_img2_sum_dim(sum2).contiguous()

        # main body
        img1, img2, sum1, sum2, img1_skip0, img2_skip0 = self.stage0(img1, img2, sum1, sum2)
        img1, img2, sum1, sum2, img1_skip1, img2_skip1 = self.stage1(img1, img2, sum1, sum2)
        
        img1, img2, sum1, sum2 = self.stage2(img1, img2, sum1, sum2, img1_skip1, img2_skip1)
        img1, img2, sum1, sum2 = self.stage3(img1, img2, sum1, sum2, img1_skip0, img2_skip0)
        
        # FIX: Ensure inputs to final stage are contiguous
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        sum1 = sum1.contiguous()
        sum2 = sum2.contiguous()
        
        output = self.stage4(img1, img2, sum1, sum2).contiguous()

        for refine_layer in self.feature_refine:
            output = (output + refine_layer(output.contiguous())).contiguous()
        
        # Final refinement
        output = (output + self.final_refine(output.contiguous())).contiguous()

        img1_attn = self.img1_spe_attn(org_img1)
        img2_attn = self.img2_spe_attn(org_img2)
        combined_attn = ((img1_attn + img2_attn) * 0.5 + 0.5).contiguous()

        output = (self.to_hrimg2(output) * combined_attn).contiguous()

        return output