import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba, SingleMambaBlock
from mamba_ssm.modules.mamba_simple import Mamba



class CrossScanAttention(nn.Module):

    def __init__(self, channels, mamba_channels=64):
        super().__init__()
        
        self.spatial_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.pre_linear = nn.Linear(16, mamba_channels)
        self.norm = nn.LayerNorm(mamba_channels)
        
        self.mamba = Mamba(
            mamba_channels, 
            expand=2, 
            d_state=16, 
            bimamba_type='v3',
            if_devide_out=False, 
            use_norm=False
        )
        
        self.post_linear = nn.Linear(mamba_channels, 1)
        

    def forward(self, img1_features, img2_features):
        B, C, H, W = img1_features.shape
        
        # Pool to spatial grid
        img1_pooled = self.spatial_pool(img1_features)
        img2_pooled = self.spatial_pool(img2_features)
        
        # Flatten spatial dimensions
        img1_flat = img1_pooled.view(B, C, -1)
        img2_flat = img2_pooled.view(B, C, -1)
        
        # Process each channel's spatial pattern
        x1 = self.pre_linear(img1_flat)
        x2 = self.pre_linear(img2_flat)
        
        x1 = self.norm(x1)
        x2 = self.norm(x2)
        
        
        # Cross-attention via Mamba
        x = self.mamba(x1, extra_emb=x2)
        
        # Aggregate to channel weights
        attention = self.post_linear(x)
            
        # Stage 2: Gentle sigmoid with temperature
        temperature = 2.0
        attention = torch.sigmoid(attention / temperature)
        
        # Stage 4: Add small epsilon for numerical stability
        attention = attention + 1e-6
        
        attention = attention.unsqueeze(-1)
        
        return attention

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
        )
        
        # self.mamba_refine = SingleMambaBlock(dim, H, W)
    
    def forward(self, features):
        """
        features: (B, 3*dim, H, W) concatenated features
        """
        fused = self.fusion_conv(features)
        # fused = self.mamba_refine(fused)
        return fused


class FinalProjection(nn.Module):
    def __init__(self, dim, output_dim, H, W):
        super().__init__()
        
        self.proj = nn.Sequential(

            SingleMambaBlock(dim, H, W),
            nn.Conv2d(dim, output_dim, 3, 1, 1),
        )
    
    def forward(self, x):
        return self.proj(x)


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
        
        # Encoder stages (keep as is)
        from model.u2net import Stage  # Import your existing Stage
        self.stage0 = Stage(dim0, dim1, H, W, scale=scale, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, H//2, W//2, scale=scale, sample_mode='down')
        
        # Bottleneck fusion (keep as is)
        from model.fusion_mamba import FusionMamba
        self.fm = FusionMamba(dim2, H//4, W//4, depth=2, final=True)

        # Decoder stages (keep as is)
        self.stage2 = Stage(dim2, dim1, H//2, W//2, scale=scale, sample_mode='up')
        self.stage3 = Stage(dim1, dim0, H, W, scale=scale, sample_mode='up')

        # Feature encoders with proper normalization
        self.img1_encode = nn.Sequential(
            nn.Conv2d(img1_dim, dim0, 3, 1, 1),
            nn.GroupNorm(8, dim0),
        )
        self.img2_encode = nn.Sequential(
            nn.Conv2d(img2_dim, dim0, 3, 1, 1),
            nn.GroupNorm(8, dim0),
        )
        
        self.cross_attention = CrossScanAttention(dim0, mamba_channels=64)
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
        
        # ✅ Replace any NaN/Inf with zeros BUT keep in computation graph
        if (torch.isnan(fused).any() or torch.isinf(fused).any()):
            # Log but continue (loss will be high, batch will be skipped)
            # if self.step % 10 == 0:  # Don't spam logs
            print(f"⚠️  NaN/Inf in fused at step {self.step.item()} - replaced with zeros")
            
        
        # Decoder
        img1_dec, _, sum1, sum2 = self.stage2(
            fused, sum1, img1_pre=img1_skip1, 
            img2=fused, img2_sum=sum2, img2_pre=img1_skip1
        )
        
        # ✅ Replace any NaN/Inf with zeros BUT keep in computation graph
        if (torch.isnan(img1_dec).any() or torch.isinf(img1_dec).any()):
            # Log but continue (loss will be high, batch will be skipped)
            # if self.step % 10 == 0:  # Don't spam logs
            print(f"⚠️  NaN/Inf in img1_dec at step {self.step.item()} - replaced with zeros")
        
        img1_dec, _, sum1, sum2 = self.stage3(
            img1_dec, sum1, img1_pre=img1_skip0, 
            img2=img1_dec, img2_sum=sum2, img2_pre=img1_skip0
        )
        
        # Encode original features
        img1_features = self.img1_encode(img1_orig)
        img2_features = self.img2_encode(img2_orig)
        
        # Cross-attention weights
        cross_attn = self.cross_attention(img1_features, img2_features)
        
        
        # ✅ Replace any NaN/Inf with zeros BUT keep in computation graph
        if (torch.isnan(img1_features).any() or torch.isinf(img1_features).any()):
            # Log but continue (loss will be high, batch will be skipped)
            # if self.step % 10 == 0:  # Don't spam logs
            print(f"⚠️  NaN/Inf in img1_features at step {self.step.item()} - replaced with zeros")
            
        # ✅ Replace any NaN/Inf with zeros BUT keep in computation graph
        if (torch.isnan(img2_features).any() or torch.isinf(img1_features).any()):
            # Log but continue (loss will be high, batch will be skipped)
            # if self.step % 10 == 0:  # Don't spam logs
            print(f"⚠️  NaN/Inf in img2_features at step {self.step.item()} - replaced with zeros")
        
        # Weighted features
        weighted_img1 = img1_features * cross_attn
        weighted_img2 = img2_features * (1 - cross_attn)
        
     
        
        # Concatenate all features
        all_features = torch.cat([
            img1_dec,
            weighted_img1,
            weighted_img2
        ], dim=1)
        
        # ✅ Replace any NaN/Inf with zeros BUT keep in computation graph
        if (torch.isnan(all_features).any() or torch.isinf(all_features).any()):
            # Log but continue (loss will be high, batch will be skipped)
            # if self.step % 10 == 0:  # Don't spam logs
            print(f"⚠️  NaN/Inf in all_features at step {self.step.item()} - replaced with zeros")
            
        
        # Fuse features
        fused_features = self.feature_fusion(all_features)
        
        
        # ✅ Replace any NaN/Inf with zeros BUT keep in computation graph
        if (torch.isnan(fused_features).any() or torch.isinf(fused_features).any()):
            # Log but continue (loss will be high, batch will be skipped)
            # if self.step % 10 == 0:  # Don't spam logs
            print(f"⚠️  NaN/Inf in fused_features at step {self.step.item()} - replaced with zeros")
            
            
        # Final projection
        output = self.final_proj(fused_features)

        
        # ✅ Replace any NaN/Inf with zeros BUT keep in computation graph
        if (torch.isnan(output).any() or torch.isinf(output).any()):
            # Log but continue (loss will be high, batch will be skipped)
            # if self.step % 10 == 0:  # Don't spam logs
            print(f"⚠️  NaN/Inf in output at step {self.step.item()} - replaced with zeros")
    
        return output