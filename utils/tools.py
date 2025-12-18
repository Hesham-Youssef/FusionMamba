import time
import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import math


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


import torch.nn.functional as F
def compute_hdr_loss(sr_log, gt_log, sr_lin, gt_lin,
                     adaptive_scale, skip_scale, epoch):

    # ---------- Base reconstruction ----------
    loss_main = F.smooth_l1_loss(sr_log, gt_log, beta=0.1)

    # ---------- Percentile matching (KEY) ----------
    percentiles = [0.90, 0.95, 0.99]
    loss_pct = 0.0
    for p in percentiles:
        sr_p = torch.quantile(sr_lin.flatten(1), p, dim=1)
        gt_p = torch.quantile(gt_lin.flatten(1), p, dim=1)
        loss_pct += F.l1_loss(sr_p, gt_p) / (gt_p.mean() + 1.0)
    loss_pct /= len(percentiles)

    # ---------- Max highlight ----------
    loss_max = F.l1_loss(
        torch.amax(sr_lin, dim=(1, 2, 3)),
        torch.amax(gt_lin, dim=(1, 2, 3))
    ) / (gt_lin.mean() + 1.0)

    # ---------- Mean + contrast ----------
    loss_mean = F.l1_loss(sr_lin.mean(), gt_lin.mean()) / (gt_lin.mean() + 1.0)

    loss_std = F.l1_loss(
        sr_lin.flatten(1).std(dim=1).mean(),
        gt_lin.flatten(1).std(dim=1).mean()
    ) / (gt_lin.std() + 1.0)

    # ---------- VERY weak scale regularization ----------
    log_ad = torch.log(adaptive_scale + 1e-6).mean()
    log_sk = torch.log(skip_scale + 1e-6).mean()

    loss_scale = 0.01 * (
        F.l1_loss(log_ad, torch.tensor(math.log(50.0), device=log_ad.device)) +
        F.l1_loss(log_sk, torch.tensor(math.log(30.0), device=log_sk.device))
    )

    # ---------- Annealed weights ----------
    w_pct = min(1.5, 0.5 + 0.1 * epoch)
    w_max = min(2.0, 0.8 + 0.15 * epoch)

    total_loss = (
        2.0 * loss_main +
        w_pct * loss_pct +
        w_max * loss_max +
        0.3 * loss_mean +
        0.3 * loss_std +
        loss_scale
    )

    return total_loss



def compute_hdr_metrics(sr_linear, gt_linear):
    """
    Compute comprehensive metrics for monitoring training progress.
    
    Args:
        sr_linear: Super-resolved output in linear space (B, C, H, W)
        gt_linear: Ground truth in linear space (B, C, H, W)
    
    Returns:
        metrics: Dictionary of computed metrics
    """
    with torch.no_grad():
        metrics = {
            # Basic statistics
            'sr_min': sr_linear.min().item(),
            'sr_max': sr_linear.max().item(),
            'sr_mean': sr_linear.mean().item(),
            'sr_std': sr_linear.std().item(),
            
            'gt_min': gt_linear.min().item(),
            'gt_max': gt_linear.max().item(),
            'gt_mean': gt_linear.mean().item(),
            'gt_std': gt_linear.std().item(),
            
            # Critical ratios
            'range_ratio': (sr_linear.max() / (gt_linear.max() + 1e-8)).item(),
            'mean_ratio': (sr_linear.mean() / (gt_linear.mean() + 1e-8)).item(),
            
            # Percentiles
            'sr_p95': torch.quantile(sr_linear.flatten(1), 0.95, dim=1).mean().item(),
            'gt_p95': torch.quantile(gt_linear.flatten(1), 0.95, dim=1).mean().item(),
            'sr_p99': torch.quantile(sr_linear.flatten(1), 0.99, dim=1).mean().item(),
            'gt_p99': torch.quantile(gt_linear.flatten(1), 0.99, dim=1).mean().item(),
        }
    
    return metrics

def reinhard_tonemap(hdr, key=0.18):
    """Reinhard tonemapping."""
    lum = 0.2126 * hdr[:, 0:1] + 0.7152 * hdr[:, 1:2] + 0.0722 * hdr[:, 2:3]
    lum_avg = torch.mean(lum, dim=[2, 3], keepdim=True)
    lum_scaled = (key / (lum_avg + 1e-6)) * lum
    tm = lum_scaled / (1.0 + lum_scaled)
    return tm.expand_as(hdr)
