import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba, SingleMambaBlock, CrossMambaBlock
from mamba_ssm.modules.mamba_simple import Mamba
from einops import rearrange


class CrossScanAttention(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.H = H
        self.W = W

        self.cross_block = CrossMambaBlock(dim, H, W)
        self.proj = nn.Linear(dim, 1)
        self.act = nn.Sigmoid()

    def forward(self, img1, img2):
        B, C, H, W = img1.shape
        
        img1 = rearrange(img1, 'b c h w -> b (h w) c')
        img2 = rearrange(img2, 'b c h w -> b (h w) c')
        fusion = self.cross_block(img1, img2)
        attn = self.act(self.proj(fusion))
        attn = rearrange(attn, 'b (h w) 1 -> b 1 h w', h=H, w=W)
        return attn


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1),
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
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

    def forward(self, x):
        return self.down(x)


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, H, W, scale=2, sample_mode='down'):
        super().__init__()        
        if sample_mode == 'down':
            self.sample = Down(in_channels, out_channels, scale)
            self.sum_sample = Down(in_channels, out_channels, scale)
        elif sample_mode == 'up':
            self.sample = Up(in_channels, out_channels, scale)
            self.sum_sample = Up(in_channels, out_channels, scale)

    def forward(self, img1, img1_sum=None, img1_pre=None, img2=None, img2_sum=None, img2_pre=None):
        if img1_pre is None:
            img1_skip = img1
            img2_skip = img2
            img1 = self.sample(img1)
            img2 = self.sample(img2)
            img1_sum = self.sum_sample(img1_sum)
            img2_sum = self.sum_sample(img2_sum)
            return img1, img2, img1_sum, img2_sum, img1_skip, img2_skip
        else:
            img1_up = self.sample(img1, img1_pre)
            img2_up = self.sample(img2, img2_pre)
            img1_sum = self.sum_sample(img1_sum, img1_pre)
            img2_sum = self.sum_sample(img2_sum, img2_pre)
            return img1_up, img2_up, img1_sum, img2_sum


class FeatureFusion(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        
        self.fusion_conv = nn.Sequential(
            nn.GroupNorm(8, dim * 3),
            nn.Conv2d(dim * 3, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
        )
    
    def forward(self, features):
        fused = self.fusion_conv(features)
        return fused


class FinalProjection(nn.Module):
    def __init__(self, dim, output_dim, H, W):
        super().__init__()
        
        self.proj = nn.Sequential(
            SingleMambaBlock(dim, H, W),
            nn.GroupNorm(8, dim),
            nn.Conv2d(dim, output_dim, 3, 1, 1),
            nn.Tanh()
        )
    
    def forward(self, x):
        output = self.proj(x)
        return output


class U2Net(nn.Module):
    def __init__(self, dim, img1_dim, img2_dim, H=64, W=64, debug=False):
        super().__init__()
        self.debug = debug
        self.register_buffer('step', torch.tensor(0))

        self.raise_img1_dim = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
        )
        self.raise_img2_dim = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
        )
        self.raise_img1_sum_dim = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
        )
        self.raise_img2_sum_dim = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
        )
        
        self.img1_dim = img1_dim
        self.img2_dim = img2_dim

        # Dimensions
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        scale = 2
        
        self.stage0 = Stage(dim0, dim1, H, W, scale=scale, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, H//2, W//2, scale=scale, sample_mode='down')
        
        self.fm = FusionMamba(dim2, H//4, W//4, depth=1, final=True)

        # Decoder stages
        self.stage2 = Stage(dim2, dim1, H//2, W//2, scale=scale, sample_mode='up')
        self.stage3 = Stage(dim1, dim0, H, W, scale=scale, sample_mode='up')

        # Feature encoders
        self.img1_encode = nn.Sequential(
            nn.Conv2d(img1_dim, dim0, 3, 1, 1),
            nn.GroupNorm(8, dim0),
        )
        self.img2_encode = nn.Sequential(
            nn.Conv2d(img2_dim, dim0, 3, 1, 1),
            nn.GroupNorm(8, dim0),
        )
        
        self.cross_attention = CrossScanAttention(dim0, H, W)
        
        self.feature_fusion = FeatureFusion(dim0, H, W)
        self.final_proj = FinalProjection(dim0, img1_dim, H, W)

    def forward(self, img1, img2, sum1, sum2):
        if self.training:
            self.step += 1
        
        # Store originals
        img1_orig = img1
        img2_orig = img2
        
        # Encode inputs
        img1 = self.raise_img1_dim(img1)
        img2 = self.raise_img2_dim(img2)
        sum1 = self.raise_img1_sum_dim(sum1)
        sum2 = self.raise_img2_sum_dim(sum2)

        # Encoder
        img1, img2, sum1, sum2, img1_skip0, img2_skip0 = self.stage0(
            img1, sum1, img2=img2, img2_sum=sum2
        )
        img1, img2, sum1, sum2, img1_skip1, img2_skip1 = self.stage1(
            img1, sum1, img2=img2, img2_sum=sum2
        )
        
        # Bottleneck fusion
        fused = self.fm(img1, img2, sum1, sum2)
        
        # Decoder
        img1_dec, _, sum1, sum2 = self.stage2(
            fused, sum1, img1_pre=img1_skip1, 
            img2=fused, img2_sum=sum2, img2_pre=img1_skip1
        )
        
        img1_dec, _, sum1, sum2 = self.stage3(
            img1_dec, sum1, img1_pre=img1_skip0, 
            img2=img1_dec, img2_sum=sum2, img2_pre=img1_skip0
        )
        
        # Encode original features
        img1_features = self.img1_encode(img1_orig)
        img2_features = self.img2_encode(img2_orig)
        
        # Cross-attention weights
        cross_attn = self.cross_attention(img1_features, img2_features)
        
        # Weighted features
        weighted_img1 = img1_features * cross_attn
        weighted_img2 = img2_features * (1 - cross_attn)
        
        # Concatenate all features
        all_features = torch.cat([
            img1_dec,
            weighted_img1,
            weighted_img2
        ], dim=1)
        
        # Fuse features
        fused_features = self.feature_fusion(all_features)
        
        # Final projection
        output = self.final_proj(fused_features)
        
        # Debug logging
        if self.training and self.step % 100 == 0:
            print(f"Step {self.step} | Output range: [{output.min():.2f}, {output.max():.2f}]")
        
        return output