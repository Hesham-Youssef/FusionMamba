import time
import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import math
import torch.nn as nn



class ERGAS(torch.nn.Module):
    def __init__(self, ratio=4):
        super().__init__()
        self.ratio = ratio

    def forward(self, img, gt):
        b, c, _, _ = img.shape
        a1 = torch.mean((img - gt) ** 2, dim=(-2, -1))
        a2 = torch.mean(gt, dim=(-2, -1)) ** 2
        com = (a1 / a2).view(b, c)
        summ = torch.sum(com, dim=-1)
        ergas = 100 * (1 / self.ratio) * ((summ / c) ** 0.5)
        ergas = ergas.mean()
        return ergas


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 3
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        #print(img1.size())
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


def gettime():
    current_time = time.localtime()
    time_str = str(current_time.tm_year) + '-' + str(current_time.tm_mon) + '-' + str(current_time.tm_mday) + \
               '-' + str(current_time.tm_hour)
    return time_str



# -----------------------------
# Optimized Tiling
# -----------------------------
from functools import lru_cache

@lru_cache(maxsize=64)
def make_hann_window(h, w, device_str='cpu'):
    """Cached Hann window with fp16 support."""
    device = torch.device(device_str)
    dtype = torch.float32
    
    wh = torch.hann_window(h, periodic=False, dtype=dtype, device=device) if h > 1 else torch.ones(1, dtype=dtype, device=device)
    ww = torch.hann_window(w, periodic=False, dtype=dtype, device=device) if w > 1 else torch.ones(1, dtype=dtype, device=device)
    return wh.unsqueeze(1) @ ww.unsqueeze(0)


def compute_tile_params(H, W, tile_h, tile_w, stride_h, stride_w):
    """Compute tiling parameters."""
    if H <= tile_h:
        pad_top, pad_bottom = 0, tile_h - H
        n_steps_h = 1
    else:
        n_steps_h = (H - tile_h + stride_h - 1) // stride_h + 1
        full_covered_h = (n_steps_h - 1) * stride_h + tile_h
        pad_top, pad_bottom = 0, max(0, full_covered_h - H)

    if W <= tile_w:
        pad_left, pad_right = 0, tile_w - W
        n_steps_w = 1
    else:
        n_steps_w = (W - tile_w + stride_w - 1) // stride_w + 1
        full_covered_w = (n_steps_w - 1) * stride_w + tile_w
        pad_left, pad_right = 0, max(0, full_covered_w - W)

    return (pad_top, pad_bottom, pad_left, pad_right), n_steps_h, n_steps_w


def extract_tiles_optimized(img, tile_h, tile_w, stride_h, stride_w, pad_mode='reflect'):
    """Ultra-optimized tile extraction using unfold."""
    C, H, W = img.shape
    
    pad_info, n_h, n_w = compute_tile_params(H, W, tile_h, tile_w, stride_h, stride_w)
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    
    # Minimize padding operations
    needs_pad = any(x > 0 for x in pad_info)
    if needs_pad:
        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode=pad_mode)
    
    # Single unfold operation for all tiles
    tiles = img.unfold(1, tile_h, stride_h).unfold(2, tile_w, stride_w)
    tiles = tiles.permute(1, 2, 0, 3, 4).reshape(-1, C, tile_h, tile_w)
    
    # Vectorized coordinate generation
    ys = torch.arange(n_h) * stride_h - pad_top
    xs = torch.arange(n_w) * stride_w - pad_left
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    
    coords = []
    for i in range(n_h * n_w):
        y, x = yy.flatten()[i].item(), xx.flatten()[i].item()
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(H, y + tile_h), min(W, x + tile_w)
        coords.append((y0, y1, x0, x1))
    
    return tiles, coords, (H, W)


def compute_summary_fast(ldr, tiles, coords, orig_shape, tile_h, tile_w):
    """Optimized summary with minimal allocations (fixed)."""
    # orig_shape is (H, W)
    H, W = orig_shape
    C = int(ldr.shape[0])
    device = ldr.device
    dtype = ldr.dtype

    # Reuse buffers
    merged = torch.zeros((C, H, W), dtype=dtype, device=device)
    weights = torch.zeros((1, H, W), dtype=dtype, device=device)

    # Ensure make_hann_window gets stable args (device as string)
    window = make_hann_window(tile_h, tile_w, device_str=str(device))

    # Batch process tiles when possible
    for idx, (y0, y1, x0, x1) in enumerate(coords):
        vh, vw = y1 - y0, x1 - x0
        w = window[:vh, :vw].unsqueeze(0)   # (1, vh, vw)
        # In-place operations: tiles[idx] has shape (C, tile_h, tile_w)
        patch = tiles[idx, :, :vh, :vw]
        merged[:, y0:y1, x0:x1].add_(patch.mul(w))
        weights[:, y0:y1, x0:x1].add_(w)

    # normalize (avoid division by zero)
    merged.div_(weights.clamp(min=1e-8))

    # Fast downsampling -> return same shape as tile (C, tile_h, tile_w)
    return F.adaptive_avg_pool2d(merged.unsqueeze(0), (tile_h, tile_w)).squeeze(0)


#!/usr/bin/env python3
"""
Unified image loading and preprocessing utilities.
Used by both training dataset and test script to ensure consistency.
"""

import os
import cv2
import numpy as np
import torch
import imageio.v2 as imageio


def _to_tensor_fast(image_np, is_hdr=False):
    """
    Optimized tensor conversion with safer HDR/LDR handling.
    
    Args:
        image_np: numpy array (H, W, C) or (H, W)
        is_hdr: If True, preserve HDR radiance values without normalization
    
    Returns:
        torch.Tensor: (C, H, W) in float32
    """
    img = image_np

    # Convert integer types to float32 with proper scaling
    if img.dtype == np.uint8:
        img = img.astype(np.float32)
        if not is_hdr:
            img = img / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32)
        if not is_hdr:
            img = img / 65535.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)

    # Debug check for HDR images with suspicious values
    if is_hdr:
        mx = float(img.max()) if hasattr(img, "max") else None
        if mx is not None and mx <= 1.5:
            print(f"WARNING: HDR image has max {mx:.4f} (<=1.5). Check if files are normalized.")

    # Ensure 3D array (add channel dim if grayscale)
    if img.ndim == 2:
        img = img[:, :, None]

    # Convert to tensor: (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    return tensor.float()


def load_image_fast(path):
    """
    Fast image loading with correct HDR/LDR handling.
    
    Args:
        path: Image file path
        
    Returns:
        numpy array: (H, W, C) in RGB order, appropriate dtype
    """
    ext = os.path.splitext(path)[1].lower()

    if ext == ".hdr":
        # Use imageio for Radiance HDR - preserves radiance values correctly
        img = imageio.imread(path).astype(np.float32)
        return img  # Already in RGB

    # LDR images - use OpenCV
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")

    # Convert BGR to RGB
    if img.ndim == 3 and img.shape[2] == 3:
        return img[:, :, ::-1].copy()

    return img


def load_triplet_fast(ldr1_path, ldr2_path, hdr_path):
    """
    Load a triplet of LDR-LDR-HDR images.
    
    Args:
        ldr1_path: Path to first LDR image
        ldr2_path: Path to second LDR image  
        hdr_path: Path to HDR ground truth
        
    Returns:
        tuple: (ldr1, ldr2, hdr) as numpy arrays in RGB
    """
    ldr1 = load_image_fast(ldr1_path)
    ldr2 = load_image_fast(ldr2_path)
    hdr = load_image_fast(hdr_path)
    
    return ldr1, ldr2, hdr


def prepare_tensors_for_inference(ldr1_np, ldr2_np, device='cuda'):
    """
    Prepare LDR images for model inference.
    
    Args:
        ldr1_np: First LDR image (numpy, RGB, 0-255 or 0-1)
        ldr2_np: Second LDR image (numpy, RGB, 0-255 or 0-1)
        device: Target device
        
    Returns:
        tuple: (t1, t2) as torch tensors (C, H, W) in [0, 1]
    """
    
    # Convert to tensors using same function as training
    t1 = _to_tensor_fast(ldr1_np, is_hdr=False).to(device)
    t2 = _to_tensor_fast(ldr2_np, is_hdr=False).to(device)
    
    return t1, t2


def prepare_tensors_for_training(ldr1_np, ldr2_np, hdr_np):
    """
    Prepare triplet for training.
    
    Args:
        ldr1_np, ldr2_np: LDR images (numpy, RGB)
        hdr_np: HDR ground truth (numpy, RGB)
        
    Returns:
        tuple: (t1, t2, tg) as torch tensors (C, H, W)
    """
    t1 = _to_tensor_fast(ldr1_np, is_hdr=False)
    t2 = _to_tensor_fast(ldr2_np, is_hdr=False)
    tg = _to_tensor_fast(hdr_np, is_hdr=True)
    
    return t1, t2, tg


class MultiScaleGradientLoss(nn.Module):
    """Multi-scale gradient loss for preserving HDR details"""
    def __init__(self, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        
    def gradient(self, x):
        # Compute gradients in x and y directions
        grad_x = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
        grad_y = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
        return grad_x, grad_y
    
    def forward(self, pred, target):
        loss = 0.0
        for scale in self.scales:
            if scale > 1:
                pred_scaled = F.avg_pool2d(pred, scale)
                target_scaled = F.avg_pool2d(target, scale)
            else:
                pred_scaled = pred
                target_scaled = target
            
            pred_grad_x, pred_grad_y = self.gradient(pred_scaled)
            target_grad_x, target_grad_y = self.gradient(target_scaled)
            
            loss += F.l1_loss(pred_grad_x, target_grad_x)
            loss += F.l1_loss(pred_grad_y, target_grad_y)
        
        return loss / len(self.scales)


class ToneMappedLoss(nn.Module):
    """Loss in tone-mapped space for perceptual quality"""
    def __init__(self, gamma=2.2):
        super().__init__()
        self.gamma = gamma
        
    def tone_map(self, x):
        # Simple Reinhard tone mapping
        # First convert from log space
        x_linear = torch.expm1(x).clamp(min=0)
        # Apply tone mapping
        x_tm = x_linear / (1.0 + x_linear)
        # Gamma correction
        x_tm = torch.pow(x_tm.clamp(min=1e-8), 1.0 / self.gamma)
        return x_tm
    
    def forward(self, pred_log, target_log):
        pred_tm = self.tone_map(pred_log)
        target_tm = self.tone_map(target_log)
        
        # L1 loss in tone-mapped space
        l1_loss = F.l1_loss(pred_tm, target_tm)
        
        # SSIM-like loss
        mu_pred = F.avg_pool2d(pred_tm, 3, 1, 1)
        mu_target = F.avg_pool2d(target_tm, 3, 1, 1)
        
        mu_pred_sq = mu_pred * mu_pred
        mu_target_sq = mu_target * mu_target
        mu_pred_target = mu_pred * mu_target
        
        sigma_pred_sq = F.avg_pool2d(pred_tm * pred_tm, 3, 1, 1) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(target_tm * target_tm, 3, 1, 1) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(pred_tm * target_tm, 3, 1, 1) - mu_pred_target
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / \
                   ((mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2))
        
        ssim_loss = 1 - ssim_map.mean()
        
        return l1_loss + 0.2 * ssim_loss


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features"""
    def __init__(self):
        super().__init__()
        # Use pre-trained VGG features
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT).features
            self.layers = nn.ModuleList([
                vgg[:4],   # relu1_2
                vgg[4:9],  # relu2_2
                vgg[9:16], # relu3_3
            ])
            for param in self.parameters():
                param.requires_grad = False
            self.use_vgg = True
        except:
            self.use_vgg = False
            print("Warning: VGG not available, perceptual loss disabled")
        
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        # Convert from log space to [0, 1]
        x = torch.expm1(x).clamp(min=0)
        x = x / (1.0 + x)  # Tone map
        # Normalize for VGG
        x = (x - self.mean) / self.std
        return x
    
    def forward(self, pred_log, target_log):
        if not self.use_vgg:
            return 0.0
        
        pred = self.normalize(pred_log)
        target = self.normalize(target_log)
        
        loss = 0.0
        for layer in self.layers:
            pred = layer(pred)
            target = layer(target)
            loss += F.l1_loss(pred, target)
        
        return loss


class EnhancedHDRLoss(nn.Module):
    """Comprehensive HDR loss combining multiple objectives"""
    def __init__(self, lambda_l1=1.0, lambda_grad=0.5, lambda_tm=0.3, lambda_perc=0.1):
        super().__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_grad = lambda_grad
        self.lambda_tm = lambda_tm
        self.lambda_perc = lambda_perc
        
        self.gradient_loss = MultiScaleGradientLoss(scales=[1, 2, 4])
        self.tonemapped_loss = ToneMappedLoss(gamma=2.2)
        self.perceptual_loss = PerceptualLoss()
    
    def forward(self, pred_log, target_log):
        # Base L1 loss in log domain
        l1_loss = F.l1_loss(pred_log, target_log)
        
        # Gradient loss for detail preservation
        grad_loss = self.gradient_loss(pred_log, target_log)
        
        # Tone-mapped perceptual loss
        tm_loss = self.tonemapped_loss(pred_log, target_log)
        
        # Perceptual loss
        perc_loss = self.perceptual_loss(pred_log, target_log)
        
        # Ensure no negative values in output
        negative_penalty = torch.relu(-pred_log).mean() * 10.0
        
        # Dynamic range preservation
        pred_range = pred_log.max() - pred_log.min()
        target_range = target_log.max() - target_log.min()
        range_loss = torch.abs(pred_range - target_range)
        
        # Combine losses
        total_loss = (self.lambda_l1 * l1_loss + 
                     self.lambda_grad * grad_loss + 
                     self.lambda_tm * tm_loss +
                     self.lambda_perc * perc_loss +
                     negative_penalty +
                     0.1 * range_loss)
        
        return total_loss, {
            'l1': l1_loss.item(),
            'grad': grad_loss.item(),
            'tm': tm_loss.item(),
            'perc': perc_loss.item() if isinstance(perc_loss, torch.Tensor) else 0.0,
            'neg_penalty': negative_penalty.item(),
            'range': range_loss.item()
        }
