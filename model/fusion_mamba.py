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
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v6', 
                           if_devide_out=False, use_norm=False, input_h=H, input_w=W)

    def forward(self, input):
        # input: (B, N, C)
        skip = input
        input = self.norm(input)
        output = self.block(input)
        # output = self.norm1(output)
        return output + skip



class CrossMambaBlock(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.norm0 = nn.LayerNorm(dim)
        self.norm1 = nn.LayerNorm(dim)
        # self.norm2 = nn.LayerNorm(dim)
        self.block = Mamba(dim, expand=1, d_state=8, bimamba_type='v7', 
                           if_devide_out=False, use_norm=False, input_h=H, input_w=W)

    def forward(self, input0, input1):
        # input0: (B, N, C) | input1: (B, N, C)
        skip = input0
        input0 = self.norm0(input0)
        input1 = self.norm1(input1)
        output = self.block(input0, extra_emb=input1)
        # output = self.norm2(output)
        return output + skip

class FusionMamba(nn.Module):
    """Enhanced FusionMamba with multi-stage fusion - FIXED for memory alignment"""
    def __init__(self, dim, H, W, depth=1, final=False):
        super().__init__()
        self.final = final
        self.spa_mamba_layers = nn.ModuleList([])
        self.spe_mamba_layers = nn.ModuleList([])
        for _ in range(depth):
            self.spa_mamba_layers.append(CrossMambaBlock(dim, H, W))
            self.spe_mamba_layers.append(CrossMambaBlock(dim, H, W))
        self.spa_cross_mamba = CrossMambaBlock(dim, H, W)
        self.spe_cross_mamba = CrossMambaBlock(dim, H, W)        
        self.out_proj = nn.Linear(dim, dim)
        
    def forward(self, img1, img2, img1_sum, img2_sum):
        b, c, h, w = img1.shape
        img1 = rearrange(img1, 'b c h w -> b (h w) c', h=h, w=w)
        img2 = rearrange(img2, 'b c h w -> b (h w) c', h=h, w=w)
        img1_sum = rearrange(img1_sum, 'b c h w -> b (h w) c', h=h, w=w)
        img2_sum = rearrange(img2_sum, 'b c h w -> b (h w) c', h=h, w=w)
        for spa_layer, spe_layer in zip(self.spa_mamba_layers, self.spe_mamba_layers):
            img1 = spa_layer(img1, img1_sum)
            img2 = spe_layer(img2, img2_sum)
            
        # Cross-modal fusion
        spa_fusion = self.spa_cross_mamba(img1, img2)
        spe_fusion = self.spe_cross_mamba(img2, img1)
        
        fusion = self.out_proj(spa_fusion + spe_fusion)
        img1 = rearrange(img1, 'b (h w) c -> b c h w', h=h, w=w)
        img2 = rearrange(img2, 'b (h w) c -> b c h w', h=h, w=w)
        img1_sum = rearrange(img1_sum, 'b (h w) c -> b c h w', h=h, w=w)
        img2_sum = rearrange(img2_sum, 'b (h w) c -> b c h w', h=h, w=w)
        output = rearrange(fusion, 'b (h w) c -> b c h w', h=h, w=w)
        if self.final:
            return output
        else:
            return (img1 + output), (img2 + output), img1_sum, img2_sum