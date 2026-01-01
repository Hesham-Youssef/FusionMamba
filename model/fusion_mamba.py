import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn


class SingleMambaBlock(nn.Module):
    """
    ✅ FIXED: Properly handles (B, C, H, W) input
    """
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.block = Mamba(
            dim, 
            expand=2, 
            d_state=16, 
            bimamba_type='v6', 
            if_devide_out=False, 
            use_norm=False, 
            input_h=H, 
            input_w=W
        )

    def forward(self, input):
        B, C, H, W = input.shape        
        skip = input
        input = rearrange(input, 'b c h w -> b (h w) c')
        input = self.norm(input)
        output = self.block(input)
        output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        return output + skip


class CrossMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=2, d_state=30, bimamba_type='v7', 
                           if_devide_out=False, use_norm=False, input_h=H, input_w=W)

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        return output + skip

class FusionMamba(nn.Module):
    def __init__(self, dim, H, W, depth=1, final=False):
        super().__init__()
        self.final = final
        
        # Self-attention layers
        self.spa_mamba_layers = nn.ModuleList([])
        self.spe_mamba_layers = nn.ModuleList([])
        self.spa_sum_mamba_layers = nn.ModuleList([])
        self.spe_sum_mamba_layers = nn.ModuleList([])
        
        for _ in range(depth):
            self.spa_mamba_layers.append(CrossMambaBlock(dim, H, W))
            self.spe_mamba_layers.append(CrossMambaBlock(dim, H, W))
            self.spa_sum_mamba_layers.append(CrossMambaBlock(dim, H, W))
            self.spe_sum_mamba_layers.append(CrossMambaBlock(dim, H, W))
        
        # Dual cross-attention: local-to-local AND local-to-global
        self.img1_cross_local = CrossMambaBlock(dim, H, W)  # img1 → img2 (local)
        self.img2_cross_local = CrossMambaBlock(dim, H, W)  # img2 → img1 (local)
        self.img1_cross_global = CrossMambaBlock(dim, H, W) # img1 → img2_sum (global)
        self.img2_cross_global = CrossMambaBlock(dim, H, W) # img2 → img1_sum (global)
        
        # Fusion projections
        self.img1_proj = nn.Linear(dim * 2, dim)  # Combine local + global
        self.img2_proj = nn.Linear(dim * 2, dim)
        
        # Final fusion
        if final:
            self.fusion_proj = nn.Sequential(
                # nn.Linear(dim * 2, dim * 2),
                # nn.GELU(),
                nn.Linear(dim * 2, dim)
            )
        
    def forward(self, img1, img2, img1_sum, img2_sum):
        b, c, h, w = img1.shape
        
        # Reshape to sequence
        img1_seq = rearrange(img1, 'b c h w -> b (h w) c')
        img2_seq = rearrange(img2, 'b c h w -> b (h w) c')
        img1_sum_seq = rearrange(img1_sum, 'b c h w -> b (h w) c')
        img2_sum_seq = rearrange(img2_sum, 'b c h w -> b (h w) c')
        
        # Store originals
        img1_orig = img1_seq
        img2_orig = img2_seq
        
        # Self-attention with own global context
        for spa_layer, spe_layer, spa_sum_layer, spe_sum_layer in zip(
            self.spa_mamba_layers, 
            self.spe_mamba_layers,
            self.spa_sum_mamba_layers,
            self.spe_sum_mamba_layers
        ):
            img1_seq = spa_layer(img1_seq, img1_sum_seq)
            img2_seq = spe_layer(img2_seq, img2_sum_seq)
            img1_sum_seq = spa_sum_layer(img1_sum_seq, img1_seq)
            img2_sum_seq = spe_sum_layer(img2_sum_seq, img2_seq)
        
        # Cross-modal fusion - BOTH local and global
        img1_local = self.img1_cross_local(img1_seq, img2_seq)       # Fine details
        img1_global = self.img1_cross_global(img1_seq, img2_sum_seq) # Global tone
        
        img2_local = self.img2_cross_local(img2_seq, img1_seq)       # Fine details
        img2_global = self.img2_cross_global(img2_seq, img1_sum_seq) # Global tone
        
        # Combine local + global information
        img1_combined = torch.cat([img1_local, img1_global], dim=-1)
        img2_combined = torch.cat([img2_local, img2_global], dim=-1)
        
        # Project with residuals
        img1_out = self.img1_proj(img1_combined) + img1_orig
        img2_out = self.img2_proj(img2_combined) + img2_orig
        
        if self.final:
            # Final fusion
            all_features = torch.cat([img1_out, img2_out], dim=-1)
            fusion = self.fusion_proj(all_features)
            return rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
        else:
            img1_out = rearrange(img1_out, 'b (h w) c -> b c h w', h=h, w=w)
            img2_out = rearrange(img2_out, 'b (h w) c -> b c h w', h=h, w=w)
            img1_sum_out = rearrange(img1_sum_seq, 'b (h w) c -> b c h w', h=h, w=w)
            img2_sum_out = rearrange(img2_sum_seq, 'b (h w) c -> b c h w', h=h, w=w)
            return img1_out, img2_out, img1_sum_out, img2_sum_out

