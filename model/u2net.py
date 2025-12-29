import torch
import torch.nn as nn
import torch.nn.functional as F
from model.fusion_mamba import FusionMamba
from mamba_ssm.modules.mamba_simple import Mamba


class CrossScanAttention(nn.Module):
    """Cross-attention between two image features (channel-wise)"""
    def __init__(self, channels, mamba_channels=64):
        super().__init__()
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        
        self.pre_linear = nn.Linear(1, mamba_channels)
        self.norm = nn.LayerNorm(mamba_channels)
        
        self.mamba = Mamba(
            mamba_channels, 
            expand=1, 
            d_state=8, 
            bimamba_type='v3',
            if_devide_out=False, 
            use_norm=True
        )
        self.post_linear = nn.Linear(mamba_channels, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, img1_features, img2_features):
        img1_pooled = self.pooling(img1_features).squeeze(-1)
        img2_pooled = self.pooling(img2_features).squeeze(-1)
        
        x1 = self.norm(self.pre_linear(img1_pooled))
        x2 = self.norm(self.pre_linear(img2_pooled))
        
        x = self.mamba(x1, extra_emb=x2)
        attention = self.post_linear(x)
        attention = attention.unsqueeze(-1)
        
        return self.sigmoid(attention)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * (scale ** 2), 3, 1, 1),
            nn.PixelShuffle(scale),
            nn.ReLU(inplace=True),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
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


class U2Net(nn.Module):
    def __init__(self, dim, img1_dim, img2_dim, H=64, W=64, debug=False):
        super().__init__()
        self.debug = debug
        self.register_buffer('step', torch.tensor(0))

        # Input projections - ADD BACK GroupNorm for stability
        self.raise_img1_dim = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True),
        )
        self.raise_img2_dim = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True),
        )
        self.raise_img1_sum_dim = nn.Sequential(
            nn.Conv2d(img1_dim, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True),
        )
        self.raise_img2_sum_dim = nn.Sequential(
            nn.Conv2d(img2_dim, dim, 3, 1, 1),
            nn.GroupNorm(8, dim),
            nn.ReLU(inplace=True),
        )
        
        self.img1_dim = img1_dim
        self.img2_dim = img2_dim

        # Dimensions
        dim0 = dim
        dim1 = int(dim0 * 2)
        dim2 = int(dim1 * 2)

        scale = 2
        
        # Encoder
        self.stage0 = Stage(dim0, dim1, H, W, scale=scale, sample_mode='down')
        self.stage1 = Stage(dim1, dim2, H//2, W//2, scale=scale, sample_mode='down')
        
        # Bottleneck fusion
        self.fm = FusionMamba(dim2, H//4, W//4, depth=2, final=True)

        # Decoder
        self.stage2 = Stage(dim2, dim1, H//2, W//2, scale=scale, sample_mode='up')
        self.stage3 = Stage(dim1, dim0, H, W, scale=scale, sample_mode='up')

        # Feature encoding - ADD BACK GroupNorm
        self.img1_encode = nn.Sequential(
            nn.Conv2d(img1_dim, dim0, 3, 1, 1),
            nn.GroupNorm(8, dim0),
        )
        self.img2_encode = nn.Sequential(
            nn.Conv2d(img2_dim, dim0, 3, 1, 1),
            nn.GroupNorm(8, dim0),
        )
        
        # Cross-attention for motion/quality assessment
        self.cross_attention = CrossScanAttention(dim0, mamba_channels=64)
        
        # Feature fusion with normalization
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(dim0 * 3, dim0 * 2, 3, 1, 1),
            nn.GroupNorm(8, dim0 * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim0 * 2, dim0, 3, 1, 1),
            nn.GroupNorm(8, dim0),
            nn.ReLU(inplace=True),
        )
        
        # CONSTRAINED output projection with tanh
        self.final_proj = nn.Sequential(
            nn.Conv2d(dim0, dim0, 3, 1, 1),
            nn.GroupNorm(8, dim0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim0, img1_dim, 3, 1, 1),
            nn.Tanh()  # Natural [-1, 1] constraint
        )
        
        # REMOVED learnable scaling - let tanh handle range
        
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights for stable training"""
        with torch.no_grad():
            # Conservative initialization for final projection
            for m in self.final_proj.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight, gain=0.1)  # Small gain for tanh
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            
            # Standard initialization for other layers
            for m in self.modules():
                if isinstance(m, nn.Conv2d) and m not in self.final_proj.modules():
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, img1, img2, sum1, sum2):
        if self.training:
            self.step += 1
        
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
        
        fused = self.fm(img1, img2, sum1, sum2)
        
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
        
        # Cross-attention weight
        cross_attn = self.cross_attention(img1_features, img2_features)
            
            
        weighted_img1 = img1_features * cross_attn
        weighted_img2 = img2_features * (1 - cross_attn)
        
        # Fuse all information
        all_features = torch.cat([
            img1_dec,
            weighted_img1,
            weighted_img2
        ], dim=1)
        
        fused_features = self.feature_fusion(all_features)
        
        # Output with natural tanh constraint
        output = self.final_proj(fused_features)
        
        # if torch.isnan(output).any():
        #     print(f"⚠️  NaN in prediction gradients")

        return output