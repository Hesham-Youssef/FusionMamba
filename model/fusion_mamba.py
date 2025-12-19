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
        self.block = Mamba(dim, expand=1, d_state=16, bimamba_type='v2', 
                           if_devide_out=True, use_norm=True)
        self.post_norm = nn.LayerNorm(dim)
        
    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        output = self.post_norm(output)
        return output + skip


class CrossMambaBlock(nn.Module):
    """Safer CrossMambaBlock that concatenates instead of using extra_emb"""
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        
        # Process concatenated features instead of using extra_emb
        self.block = Mamba(dim * 2, expand=1, d_state=16, bimamba_type='v2', 
                           if_devide_out=True, use_norm=True)
        
        self.post_norm = nn.LayerNorm(dim * 2)
        
        # Project back to original dimension
        self.proj_back = nn.Linear(dim * 2, dim)
        
        # Cross-attention weight
        self.cross_weight = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )

    def forward(self, input0, input1):
        B, N, C = input0.shape
        
        # Ensure contiguous and validate
        input0 = input0.contiguous()
        input1 = input1.contiguous()
        
        # Check for NaN/Inf
        if torch.isnan(input0).any() or torch.isinf(input0).any():
            input0 = torch.nan_to_num(input0, nan=0.0, posinf=1.0, neginf=-1.0)
        if torch.isnan(input1).any() or torch.isinf(input1).any():
            input1 = torch.nan_to_num(input1, nan=0.0, posinf=1.0, neginf=-1.0)
        
        skip = input0
        
        # Normalize
        input0_norm = self.norm0(input0)
        input1_norm = self.norm1(input1)
        
        # Concatenate instead of using extra_emb (safer approach)
        combined = torch.cat([input0_norm, input1_norm], dim=-1)
        
        # Compute attention weight
        weight = self.cross_weight(combined)
        weight = torch.clamp(weight, min=0.01, max=0.99)
        
        # Process through Mamba (no extra_emb parameter)
        output = self.block(combined)
        output = self.post_norm(output)
        
        # Project back to original dimension
        output = self.proj_back(output)
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply weighted fusion
        output = output * weight + input0_norm * (1 - weight)
        
        return output + skip


class FusionMamba(nn.Module):
    """Safer FusionMamba without extra_emb parameter"""
    def __init__(self, dim, H, W, depth=2, final=False):
        super().__init__()
        self.final = final
        self.depth = depth
        self.H = H
        self.W = W
        
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
        
        # Output projection
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Learnable fusion weights
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, img1, img2, img1_sum, img2_sum):
        b, c, h, w = img1.shape
        
        # Clean any NaN/Inf in inputs
        img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=-1.0)
        img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=-1.0)
        img1_sum = torch.nan_to_num(img1_sum, nan=0.0, posinf=1.0, neginf=-1.0)
        img2_sum = torch.nan_to_num(img2_sum, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
        
        # Weighted combination
        alpha = torch.clamp(self.alpha, min=0.1, max=2.0)
        beta = torch.clamp(self.beta, min=0.1, max=2.0)
        
        fusion = alpha * spa_fusion + beta * spe_fusion
        fusion = fusion / (alpha + beta + 1e-6)
        
        # Self-refinement
        fusion = self.self_attn(fusion)
        
        # Final projection
        fusion = self.out_proj(fusion)
        fusion = torch.nan_to_num(fusion, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Reshape back
        img1 = rearrange(img1, 'b (h w) c -> b c h w', h=h, w=w)
        img2 = rearrange(img2, 'b (h w) c -> b c h w', h=h, w=w)
        output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
        
        if self.final:
            return output
        else:
            result1 = (img1 + output) * 0.5
            result2 = (img2 + output) * 0.5
            return result1, result2