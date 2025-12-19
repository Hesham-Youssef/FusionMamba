import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn


class SingleMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=16, bimamba_type='v6', 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)
        self.post_norm = nn.LayerNorm(dim)
        
    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        output = self.post_norm(output)
        return output + skip


class CrossMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=16, bimamba_type='v7', 
                           if_devide_out=True, use_norm=True, input_h=H, input_w=W)
        self.post_norm = nn.LayerNorm(dim)
        # Add cross-attention weighting
        self.cross_weight = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        
        input0_norm = self.norm0(input0)
        input1_norm = self.norm1(input1)
        
        # Compute cross-attention weight
        combined = torch.cat([input0_norm, input1_norm], dim=-1)
        weight = self.cross_weight(combined)
        
        # Apply weighted fusion
        output = self.block(input0_norm, extra_emb=input1_norm)
        output = self.post_norm(output)
        output = output * weight + input0_norm * (1 - weight)
        
        return output + skip

class FusionMamba(nn.Module):
    """Enhanced FusionMamba with multi-stage fusion"""
    def __init__(self, dim, H, W, depth=2, final=False):
        super().__init__()
        self.final = final
        self.depth = depth
        
        # Spatial and spectral processing layers
        self.spa_mamba_layers = nn.ModuleList([])
        self.spe_mamba_layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.spa_mamba_layers.append(CrossMambaBlock(dim, H, W))
            self.spe_mamba_layers.append(CrossMambaBlock(dim, H, W))
        
        # Cross-fusion layers
        self.spa_cross_mamba = CrossMambaBlock(dim, H, W)
        self.spe_cross_mamba = CrossMambaBlock(dim, H, W)
        
        # Self-attention for refinement
        self.self_attn = SingleMambaBlock(dim, H, W)
        
        # Output projection with gating
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        
    def forward(self, img1, img2, img1_sum, img2_sum):
        b, c, h, w = img1.shape
        
        # Reshape to sequence format
        img1 = rearrange(img1, 'b c h w -> b (h w) c', h=h, w=w)
        img2 = rearrange(img2, 'b c h w -> b (h w) c', h=h, w=w)
        img1_sum = rearrange(img1_sum, 'b c h w -> b (h w) c', h=h, w=w)
        img2_sum = rearrange(img2_sum, 'b c h w -> b (h w) c', h=h, w=w)
        
        # Progressive fusion through layers
        for spa_layer, spe_layer in zip(self.spa_mamba_layers, self.spe_mamba_layers):
            img1 = spa_layer(img1, img1_sum)
            img2 = spe_layer(img2, img2_sum)
        
        # Cross-modal fusion
        spa_fusion = self.spa_cross_mamba(img1, img2)
        spe_fusion = self.spe_cross_mamba(img2, img1)
        
        # Weighted combination with learnable parameters
        fusion = self.alpha * spa_fusion + self.beta * spe_fusion
        fusion = fusion / (self.alpha + self.beta)  # Normalize
        
        # Self-refinement
        fusion = self.self_attn(fusion)
        
        # Final projection
        fusion = self.out_proj(fusion)
        
        # Reshape back
        img1 = rearrange(img1, 'b (h w) c -> b c h w', h=h, w=w)
        img2 = rearrange(img2, 'b (h w) c -> b c h w', h=h, w=w)
        output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
        
        if self.final:
            return output
        else:
            # Residual connections for stability
            return (img1 + output) * 0.5, (img2 + output) * 0.5


