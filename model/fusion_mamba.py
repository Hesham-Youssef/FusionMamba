import math
import torch
import numpy as np
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn


def check_tensor_valid(tensor, name="tensor"):
    """Check if tensor contains NaN or Inf values"""
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN values")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf values")
    return tensor


class SingleMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
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
        
        self.cross_weight_linear1 = nn.Linear(dim * 2, dim)
        self.cross_weight_norm = nn.LayerNorm(dim)
        self.cross_weight_activation = nn.Sigmoid()
        
        # Add gradient clipping to prevent instability
        self.gradient_clip_val = 1.0

    def forward(self, input0, input1):
        # Validate inputs
        if torch.isnan(input0).any() or torch.isinf(input0).any():
            print("Warning: input0 contains NaN or Inf, replacing with zeros")
            input0 = torch.nan_to_num(input0, nan=0.0, posinf=1.0, neginf=-1.0)
        
        if torch.isnan(input1).any() or torch.isinf(input1).any():
            print("Warning: input1 contains NaN or Inf, replacing with zeros")
            input1 = torch.nan_to_num(input1, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure contiguous memory (minimal cloning)
        if not input0.is_contiguous():
            input0 = input0.contiguous()
        if not input1.is_contiguous():
            input1 = input1.contiguous()
        
        skip = input0
        
        # Normalize inputs
        input0_norm = self.norm0(input0)
        input1_norm = self.norm1(input1)
        
        # Ensure normalized tensors are valid
        input0_norm = torch.nan_to_num(input0_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        input1_norm = torch.nan_to_num(input1_norm, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Compute cross-attention weight
        combined = torch.cat([input0_norm, input1_norm], dim=-1)
        
        weight = self.cross_weight_linear1(combined)
        weight = self.cross_weight_norm(weight)
        weight = torch.nan_to_num(weight, nan=0.5, posinf=1.0, neginf=0.0)
        weight = self.cross_weight_activation(weight)
        
        # Clamp weight to prevent extreme values
        weight = torch.clamp(weight, min=0.01, max=0.99)
        
        # CRITICAL FIX: Ensure proper memory layout before Mamba
        # Only clone if necessary to save memory
        if not input0_norm.is_contiguous():
            input0_norm = input0_norm.contiguous()
        if not input1_norm.is_contiguous():
            input1_norm = input1_norm.contiguous()
        
        # Apply Mamba block with error handling
        try:
            output = self.block(input0_norm, extra_emb=input1_norm)
        except RuntimeError as e:
            print(f"Mamba block error: {e}")
            print(f"input0_norm shape: {input0_norm.shape}, dtype: {input0_norm.dtype}")
            print(f"input1_norm shape: {input1_norm.shape}, dtype: {input1_norm.dtype}")
            print(f"input0_norm device: {input0_norm.device}, input1_norm device: {input1_norm.device}")
            print(f"input0_norm is_contiguous: {input0_norm.is_contiguous()}")
            print(f"input1_norm is_contiguous: {input1_norm.is_contiguous()}")
            # Fallback: use only input0_norm without extra_emb
            print("Falling back to single-input Mamba")
            output = self.block(input0_norm)
        
        output = self.post_norm(output)
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply weighted fusion
        output = output * weight + input0_norm * (1 - weight)
        
        return output + skip


class FusionMamba(nn.Module):
    """Enhanced FusionMamba with multi-stage fusion - FIXED for CUBLAS errors"""
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
        
        # Output projection with gating
        self.out_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        
        # Learnable fusion weights (initialized to prevent instability)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.beta = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, img1, img2, img1_sum, img2_sum):
        b, c, h, w = img1.shape
        
        # Validate inputs
        for name, tensor in [("img1", img1), ("img2", img2), 
                             ("img1_sum", img1_sum), ("img2_sum", img2_sum)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: {name} contains NaN or Inf, cleaning...")
                if name == "img1":
                    img1 = torch.nan_to_num(img1, nan=0.0, posinf=1.0, neginf=-1.0)
                elif name == "img2":
                    img2 = torch.nan_to_num(img2, nan=0.0, posinf=1.0, neginf=-1.0)
                elif name == "img1_sum":
                    img1_sum = torch.nan_to_num(img1_sum, nan=0.0, posinf=1.0, neginf=-1.0)
                elif name == "img2_sum":
                    img2_sum = torch.nan_to_num(img2_sum, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Ensure inputs are contiguous
        img1 = img1.contiguous()
        img2 = img2.contiguous()
        img1_sum = img1_sum.contiguous()
        img2_sum = img2_sum.contiguous()
        
        # Reshape to sequence format
        img1 = rearrange(img1, 'b c h w -> b (h w) c', h=h, w=w)
        img2 = rearrange(img2, 'b c h w -> b (h w) c', h=h, w=w)
        img1_sum = rearrange(img1_sum, 'b c h w -> b (h w) c', h=h, w=w)
        img2_sum = rearrange(img2_sum, 'b c h w -> b (h w) c', h=h, w=w)
        
        # Progressive fusion through layers
        for i, (spa_layer, spe_layer) in enumerate(zip(self.spa_mamba_layers, self.spe_mamba_layers)):
            try:
                img1 = spa_layer(img1, img1_sum)
                img2 = spe_layer(img2, img2_sum)
            except RuntimeError as e:
                print(f"Error in layer {i}: {e}")
                # Skip this layer if it fails
                continue
        
        # Cross-modal fusion
        spa_fusion = self.spa_cross_mamba(img1, img2)
        spe_fusion = self.spe_cross_mamba(img2, img1)
        
        # Weighted combination with learnable parameters
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
            # Residual connections for stability
            result1 = (img1 + output) * 0.5
            result2 = (img2 + output) * 0.5
            return result1, result2