#!/usr/bin/env python3
# tester_tiled_optimized.py
"""
Optimized patch-based tiled tester with:
 - Batch processing for speed
 - Lazy tile generation for memory efficiency
 - Mixed precision support
 - Efficient summary computation
"""

import os
import argparse
from pathlib import Path

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext

# --- Utilities ---
def _to_tensor_and_normalize(image_np, is_hdr=False):
    img = image_np.astype(np.float32)
    if not is_hdr:
        if img.max() > 1.5:
            img = img / 255.0
    if img.ndim == 2:
        img = img[:, :, None]
    img_t = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_t

def make_hann_window(h, w, device=None, dtype=torch.float32):
    if h == 1:
        wh = torch.tensor([1.0], dtype=dtype, device=device)
    else:
        wh = torch.hann_window(h, periodic=False, dtype=dtype, device=device)
    if w == 1:
        ww = torch.tensor([1.0], dtype=dtype, device=device)
    else:
        ww = torch.hann_window(w, periodic=False, dtype=dtype, device=device)
    return wh.unsqueeze(1) @ ww.unsqueeze(0)

class TileGenerator:
    """Memory-efficient tile generator that yields tiles on-the-fly"""
    def __init__(self, img, tile_h, tile_w, stride_h, stride_w, pad_mode='reflect'):
        self.img = img
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.stride_h = stride_h
        self.stride_w = stride_w
        self.pad_mode = pad_mode
        
        C, H, W = img.shape
        self.orig_shape = (H, W)
        
        # Compute padding
        if H <= tile_h:
            pad_top, pad_bottom = 0, tile_h - H
        else:
            n_steps_h = math.ceil((H - tile_h) / float(stride_h)) + 1
            full_covered_h = (n_steps_h - 1) * stride_h + tile_h
            pad_top, pad_bottom = 0, max(0, full_covered_h - H)

        if W <= tile_w:
            pad_left, pad_right = 0, tile_w - W
        else:
            n_steps_w = math.ceil((W - tile_w) / float(stride_w)) + 1
            full_covered_w = (n_steps_w - 1) * stride_w + tile_w
            pad_left, pad_right = 0, max(0, full_covered_w - W)

        self.pads = (pad_top, pad_bottom, pad_left, pad_right)
        
        if any(x > 0 for x in self.pads):
            self.img_p = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode=pad_mode)
        else:
            self.img_p = img
            
        _, self.Hp, self.Wp = self.img_p.shape
        
    def __iter__(self):
        """Yields (tile, coord) tuples"""
        H, W = self.orig_shape
        pad_top, _, pad_left, _ = self.pads
        
        for y in range(0, self.Hp - self.tile_h + 1, self.stride_h):
            for x in range(0, self.Wp - self.tile_w + 1, self.stride_w):
                tile = self.img_p[:, y:y + self.tile_h, x:x + self.tile_w]
                y0 = max(0, y - pad_top)
                x0 = max(0, x - pad_left)
                y1 = min(H, y - pad_top + self.tile_h)
                x1 = min(W, x - pad_left + self.tile_w)
                yield tile, (y0, y1, x0, x1)
    
    def get_all_coords(self):
        """Get all coordinates without extracting tiles"""
        H, W = self.orig_shape
        pad_top, _, pad_left, _ = self.pads
        coords = []
        
        for y in range(0, self.Hp - self.tile_h + 1, self.stride_h):
            for x in range(0, self.Wp - self.tile_w + 1, self.stride_w):
                y0 = max(0, y - pad_top)
                x0 = max(0, x - pad_left)
                y1 = min(H, y - pad_top + self.tile_h)
                x1 = min(W, x - pad_left + self.tile_w)
                coords.append((y0, y1, x0, x1))
        return coords


def compute_summary_efficient(ldr1, ldr2, tile_h, tile_w, stride_h, stride_w, device):
    """Efficient summary computation using streaming approach"""
    H, W = ldr1.shape[1], ldr1.shape[2]
    
    # Create generators
    gen1 = TileGenerator(ldr1, tile_h, tile_w, stride_h, stride_w)
    gen2 = TileGenerator(ldr2, tile_h, tile_w, stride_h, stride_w)
    
    # Merge with Hann weighting (streaming)
    win = make_hann_window(tile_h, tile_w, device=device, dtype=ldr1.dtype)
    
    out1 = torch.zeros_like(ldr1)
    out2 = torch.zeros_like(ldr2)
    wgt = torch.zeros((1, H, W), dtype=ldr1.dtype, device=device)
    
    for (tile1, coord), (tile2, _) in zip(gen1, gen2):
        y0, y1, x0, x1 = coord
        vh, vw = y1 - y0, x1 - x0
        ws = win[:vh, :vw].unsqueeze(0)
        
        out1[:, y0:y1, x0:x1] += tile1[:, :vh, :vw] * ws
        out2[:, y0:y1, x0:x1] += tile2[:, :vh, :vw] * ws
        wgt[:, y0:y1, x0:x1] += ws
    
    out1 = out1 / (wgt + 1e-8)
    out2 = out2 / (wgt + 1e-8)
    
    # Downsample to summary size
    sum1 = F.adaptive_avg_pool2d(out1.unsqueeze(0), (tile_h, tile_w)).squeeze(0)
    sum2 = F.adaptive_avg_pool2d(out2.unsqueeze(0), (tile_h, tile_w)).squeeze(0)
    
    return sum1, sum2


# Original methods for verification
def extract_tiles_old(img, tile_h, tile_w, stride_h, stride_w, pad_mode='reflect'):
    """Original tile extraction that returns all tiles as a list"""
    C, H, W = img.shape
    
    if H <= tile_h:
        pad_top, pad_bottom = 0, tile_h - H
    else:
        n_steps_h = math.ceil((H - tile_h) / float(stride_h)) + 1
        full_covered_h = (n_steps_h - 1) * stride_h + tile_h
        pad_top, pad_bottom = 0, max(0, full_covered_h - H)

    if W <= tile_w:
        pad_left, pad_right = 0, tile_w - W
    else:
        n_steps_w = math.ceil((W - tile_w) / float(stride_w)) + 1
        full_covered_w = (n_steps_w - 1) * stride_w + tile_w
        pad_left, pad_right = 0, max(0, full_covered_w - W)

    if any(x > 0 for x in (pad_top, pad_bottom, pad_left, pad_right)):
        img_p = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode=pad_mode)
    else:
        img_p = img

    _, Hp, Wp = img_p.shape
    tiles, coords = [], []
    for y in range(0, Hp - tile_h + 1, stride_h):
        for x in range(0, Wp - tile_w + 1, stride_w):
            tile = img_p[:, y:y + tile_h, x:x + tile_w]
            y0 = max(0, y - pad_top)
            x0 = max(0, x - pad_left)
            y1 = min(H, y - pad_top + tile_h)
            x1 = min(W, x - pad_left + tile_w)
            tiles.append(tile)
            coords.append((y0, y1, x0, x1))
    return tiles, coords, (H, W), (pad_top, pad_bottom, pad_left, pad_right)


def compute_summary_old(ldr1, ldr2, tiles1, tiles2, coords, orig_shape, tile_h, tile_w, device):
    """Original summary computation"""
    H, W = orig_shape
    dtype = ldr1.dtype
    win = make_hann_window(tile_h, tile_w, device=device, dtype=dtype)

    def merge(tiles, base):
        out = torch.zeros_like(base)
        wgt = torch.zeros((1, H, W), dtype=base.dtype, device=device)
        for tile, (y0, y1, x0, x1) in zip(tiles, coords):
            vh, vw = y1 - y0, x1 - x0
            ws = win[:vh, :vw].unsqueeze(0)
            out[:, y0:y1, x0:x1] += tile[:, :vh, :vw] * ws
            wgt[:, y0:y1, x0:x1] += ws
        return out / (wgt + 1e-8)

    s1 = merge(tiles1, ldr1)
    s2 = merge(tiles2, ldr2)

    s1_pool = F.adaptive_avg_pool2d(s1.unsqueeze(0), (tile_h, tile_w)).squeeze(0)
    s2_pool = F.adaptive_avg_pool2d(s2.unsqueeze(0), (tile_h, tile_w)).squeeze(0)
    return s1_pool, s2_pool


def compute_metrics(sr, gt):
    from pytorch_msssim import ssim
    sr = sr.clamp(0, 1)
    gt = gt.clamp(0, 1)
    ssim_val = ssim(sr, gt, data_range=1.0, size_average=True).item()
    mse = ((sr - gt) ** 2).mean()
    psnr_val = 10 * torch.log10(1.0 / (mse + 1e-12)).item()
    return ssim_val, psnr_val


def test(args):
    # Model load
    from model.u2net import U2Net as Net
    model = Net(
        args.channels,
        args.spa_channels,
        args.spe_channels,
        args.cut_size + 2 * args.pad,
        args.cut_size + 2 * args.pad,
        args.ratio
    ).to(args.device)

    state_dict = torch.load(args.weight, map_location=args.device)
    if isinstance(state_dict, dict):
        if 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    new_state = {}
    for k, v in state_dict.items():
        new_state[k.replace('module.', '')] = v
    model.load_state_dict(new_state)
    model.eval()
    
    # Enable mixed precision if requested
    use_amp = args.use_amp and args.device == 'cuda'
    if use_amp:
        print("WARNING: Using automatic mixed precision (AMP) - disable with --no-use_amp if results are poor")
    
    if args.batch_size > 1:
        print(f"Using batch processing with batch_size={args.batch_size}")
    else:
        print("Using single-tile processing (slower but safer)")

    os.makedirs(args.save_dir, exist_ok=True)

    scenes = sorted(
        d for d in os.listdir(args.test_data_path)
        if os.path.isdir(os.path.join(args.test_data_path, d))
    )

    for idx, scene in enumerate(scenes):
        scene_path = os.path.join(args.test_data_path, scene)

        ldr_files = sorted(
            f for f in os.listdir(scene_path)
            if f.startswith("input_") and f.endswith("_aligned.tif")
        )
        hdr_files = sorted(
            f for f in os.listdir(scene_path)
            if f.startswith("ref_hdr") and f.endswith("_aligned.hdr")
        )
        if len(ldr_files) < 2 or len(hdr_files) == 0:
            continue

        # Read images
        ldr_long = cv2.imread(os.path.join(scene_path, ldr_files[0]), -1).astype(np.float32) / 255.0
        ldr_short = cv2.imread(os.path.join(scene_path, ldr_files[1]), -1).astype(np.float32) / 255.0
        gt_hdr = cv2.imread(os.path.join(scene_path, hdr_files[0]), -1).astype(np.float32)

        if ldr_long.ndim == 3:
            ldr_long = cv2.cvtColor(ldr_long, cv2.COLOR_BGR2RGB)
            ldr_short = cv2.cvtColor(ldr_short, cv2.COLOR_BGR2RGB)
        if gt_hdr.ndim == 3:
            gt_hdr = cv2.cvtColor(gt_hdr, cv2.COLOR_BGR2RGB)

        # To tensors
        t1 = _to_tensor_and_normalize(ldr_long).to(args.device)
        t2 = _to_tensor_and_normalize(ldr_short).to(args.device)

        H, W = t1.shape[1], t1.shape[2]

        # Compute summaries efficiently
        cut = args.cut_size
        stride = cut // 2

        if args.verify_summaries:
            # Use original method for verification
            print("  [VERIFY] Computing summaries with original method...")
            tiles1_list, coords, orig_shape, _ = extract_tiles_old(t1, cut, cut, stride, stride)
            tiles2_list, _, _, _ = extract_tiles_old(t2, cut, cut, stride, stride)
            sum1_orig, sum2_orig = compute_summary_old(
                t1, t2, tiles1_list, tiles2_list, coords, orig_shape, cut, cut, args.device
            )
            
            print("  [VERIFY] Computing summaries with optimized method...")
            sum1_opt, sum2_opt = compute_summary_efficient(
                t1, t2, cut, cut, stride, stride, args.device
            )
            
            diff1 = (sum1_orig - sum1_opt).abs().mean()
            diff2 = (sum2_orig - sum2_opt).abs().mean()
            print(f"  [VERIFY] Summary difference: sum1={diff1:.6f}, sum2={diff2:.6f}")
            
            if diff1 > 1e-5 or diff2 > 1e-5:
                print("  [WARNING] Summaries differ! Using original method.")
                sum1, sum2 = sum1_orig, sum2_orig
            else:
                print("  [OK] Summaries match. Using optimized method.")
                sum1, sum2 = sum1_opt, sum2_opt
        else:
            sum1, sum2 = compute_summary_efficient(
                t1, t2, cut, cut, stride, stride, args.device
            )
        
        # Prepare for tiled inference
        pad = args.pad
        ratio = args.ratio
        ms_size = cut // ratio

        ldr_long_t = torch.from_numpy(ldr_long).permute(2, 0, 1).unsqueeze(0).to(args.device)
        ldr_short_t = torch.from_numpy(ldr_short).permute(2, 0, 1).unsqueeze(0).to(args.device)

        ldr_long_pad = F.pad(ldr_long_t, (pad, pad, pad, pad), 'reflect')
        ldr_short_pad = F.pad(
            ldr_short_t,
            (pad // ratio, pad // ratio, pad // ratio, pad // ratio),
            'reflect'
        )

        edge_H = (cut - (H % cut)) % cut
        edge_W = (cut - (W % cut)) % cut

        ldr_long_pad = F.pad(ldr_long_pad, (0, edge_W, 0, edge_H), 'reflect')
        ldr_short_pad = F.pad(
            ldr_short_pad,
            (0, edge_W // ratio, 0, edge_H // ratio),
            'reflect'
        )

        H_pad, W_pad = ldr_long_pad.shape[2:]
        C = ldr_long_pad.shape[1]

        # Accumulators
        out_acc = torch.zeros((1, C, H_pad, W_pad), device=args.device)
        w_acc = torch.zeros((1, 1, H_pad, W_pad), device=args.device)

        win = make_hann_window(cut, cut, device=args.device).unsqueeze(0)

        sum1_b = sum1.unsqueeze(0)
        sum2_b = sum2.unsqueeze(0)

        # Collect tile positions
        tile_positions = []
        for y in range(0, H_pad - cut + 1, stride):
            for x in range(0, W_pad - cut + 1, stride):
                tile_positions.append((y, x))

        # Batch processing
        batch_size = args.batch_size
        
        # Track raw model outputs before any processing
        if args.save_debug:
            raw_outputs_sample = []
        
        with torch.no_grad():
            autocast_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
            
            for batch_start in range(0, len(tile_positions), batch_size):
                batch_end = min(batch_start + batch_size, len(tile_positions))
                batch_tiles = tile_positions[batch_start:batch_end]
                
                # Prepare batch
                MS_batch = []
                PAN_batch = []
                
                for y, x in batch_tiles:
                    MS_patch = ldr_short_pad[
                        :, :,
                        y // ratio : y // ratio + ms_size + pad // 2,
                        x // ratio : x // ratio + ms_size + pad // 2
                    ]
                    PAN_patch = ldr_long_pad[
                        :, :,
                        y : y + cut + 2 * pad,
                        x : x + cut + 2 * pad
                    ]
                    MS_batch.append(MS_patch)
                    PAN_batch.append(PAN_patch)
                
                MS_batch = torch.cat(MS_batch, dim=0)
                PAN_batch = torch.cat(PAN_batch, dim=0)
                
                # Replicate summaries for batch
                sum1_batch = sum1_b.expand(len(batch_tiles), -1, -1, -1)
                sum2_batch = sum2_b.expand(len(batch_tiles), -1, -1, -1)
                
                # Inference with mixed precision
                with autocast_context:
                    sr_batch = model(MS_batch, PAN_batch, sum1_batch, sum2_batch)
                
                # CRITICAL: Save raw model output BEFORE any clipping/processing
                if args.save_debug and batch_start == 0:
                    raw_outputs_sample.append(sr_batch[0].detach().clone())
                
                # DON'T clamp here if using log-space! Let values exceed 1.0
                if args.use_log:
                    # In log space, values can exceed 1.0 (log(1+large_HDR) > 1)
                    sr_batch = sr_batch.float().clamp(min=0)  # Only clamp negative values
                else:
                    # In linear space, clamp to [0, 1]
                    sr_batch = sr_batch.float().clamp(0, 1)
                
                # Accumulate results
                for i, (y, x) in enumerate(batch_tiles):
                    sr_center = sr_batch[i:i+1, :, pad:pad + cut, pad:pad + cut]
                    out_acc[:, :, y:y + cut, x:x + cut] += sr_center * win
                    w_acc[:, :, y:y + cut, x:x + cut] += win
                
                # Clear batch from GPU memory
                del MS_batch, PAN_batch, sum1_batch, sum2_batch, sr_batch
                if args.device == 'cuda':
                    torch.cuda.empty_cache()

        out_final = out_acc / (w_acc + 1e-8)
        out_final = out_final[:, :, :H, :W]

        # Save outputs
        output_np = out_final.squeeze(0).permute(1, 2, 0).cpu().numpy()
        
        # ============================================================
        # DIAGNOSTIC: Analyze where clipping happens
        # ============================================================
        if args.save_debug:
            print(f'\n  [DIAGNOSTIC] Clipping Analysis for Sample {idx+1}:')
            print(f'  {"="*60}')
            
            # 1. Raw model output (first tile, before any processing)
            if raw_outputs_sample:
                raw_first = raw_outputs_sample[0].cpu().numpy()
                print(f'  1. Raw model output (first tile):')
                print(f'     Range: [{raw_first.min():.4f}, {raw_first.max():.4f}]')
                print(f'     Mean: {raw_first.mean():.4f}, Std: {raw_first.std():.4f}')
                print(f'     Values > 1.0: {(raw_first > 1.0).sum()} pixels')
                print(f'     Values > 0.5: {(raw_first > 0.5).sum()} pixels')
            
            # 2. After merging but before log inverse
            print(f'  2. After tile merging (before expm1):')
            print(f'     Range: [{out_final.min().item():.4f}, {out_final.max().item():.4f}]')
            print(f'     Mean: {out_final.mean().item():.4f}')
            print(f'     Values > 1.0: {(out_final > 1.0).sum().item()} pixels')
            print(f'     Values > 0.5: {(out_final > 0.5).sum().item()} pixels')

        if args.use_log:
            output_np = np.expm1(output_np)
        output_np = np.clip(output_np, 0, None)  # Only clip negatives, allow large HDR values

        # Save HDR
        cv2.imwrite(
            os.path.join(args.save_dir, f'output_{idx+1}.hdr'),
            output_np[..., ::-1].astype(np.float32)
        )
        
        # Print HDR statistics for debugging
        if args.save_debug:
            print(f'  3. After expm1 (final HDR):')
            print(f'     Range: [{output_np.min():.4f}, {output_np.max():.4f}]')
            print(f'     Mean: {output_np.mean():.4f}')
            print(f'     Values > 1.0: {(output_np > 1.0).sum()} pixels')
            print(f'     Values > 0.5: {(output_np > 0.5).sum()} pixels')
            print(f'  4. Ground truth HDR:')
            print(f'     Range: [{gt_hdr.min():.4f}, {gt_hdr.max():.4f}]')
            print(f'     Mean: {gt_hdr.mean():.4f}')
            print(f'     Values > 1.0: {(gt_hdr > 1.0).sum()} pixels')
            print(f'     Values > 0.5: {(gt_hdr > 0.5).sum()} pixels')
            
            # Check if model architecture has clipping
            print(f'\n  [ANALYSIS] Likely cause of clipping:')
            if raw_outputs_sample and raw_first.max() < 0.45:
                print(f'     ⚠️  MODEL ARCHITECTURE ISSUE: Raw output max={raw_first.max():.4f}')
                print(f'     The model itself is not predicting high values!')
                print(f'     Check: activation functions (sigmoid/tanh), training data range')
            elif raw_outputs_sample and raw_first.max() > 0.9:
                print(f'     ✓ Model outputs high values (max={raw_first.max():.4f})')
                if out_final.max().item() < 0.45:
                    print(f'     ⚠️  TESTING ROUTINE ISSUE: Values clipped during merging')
                else:
                    print(f'     ✓ Testing routine preserves high values')
            else:
                print(f'     → Model outputs moderate values (max={raw_first.max() if raw_outputs_sample else "N/A"})')
            print(f'  {"="*60}\n')

        # ============================================================
        # METRICS: Use original tone mapping (Reinhard, no exposure adjustment)
        # ============================================================
        sr_tm_for_metrics = output_np / (1.0 + output_np)
        sr_tm_for_metrics = np.clip(sr_tm_for_metrics, 0, 1)
        
        gt_tm_for_metrics = gt_hdr / (1.0 + gt_hdr)
        gt_tm_for_metrics = np.clip(gt_tm_for_metrics, 0, 1)
        
        sr_tm_tensor = torch.from_numpy(sr_tm_for_metrics).permute(2, 0, 1).unsqueeze(0).to(args.device)
        gt_tm_tensor = torch.from_numpy(gt_tm_for_metrics).permute(2, 0, 1).unsqueeze(0).to(args.device)
        
        ssim_val, psnr_val = compute_metrics(sr_tm_tensor, gt_tm_tensor)
        
        # ============================================================
        # VISUALIZATION: Apply exposure and better tone mapping for PNG
        # ============================================================
        if args.auto_exposure:
            percentile_val = np.percentile(output_np, 95)
            if percentile_val > 0:
                exposure_scale = args.target_brightness / percentile_val
                output_exposed = output_np * exposure_scale
            else:
                output_exposed = output_np
        else:
            output_exposed = output_np * args.exposure
        
        # Apply tone mapping for visualization
        if args.tonemap_method == 'reinhard':
            tm = output_exposed / (1.0 + output_exposed)
        elif args.tonemap_method == 'gamma':
            normalized = np.clip(output_exposed / np.percentile(output_exposed, 99), 0, 1)
            tm = np.power(normalized, 1.0/args.gamma)
        elif args.tonemap_method == 'log':
            tm = np.log(1 + output_exposed * args.log_scale) / np.log(1 + args.log_scale)
        elif args.tonemap_method == 'aces':
            a = 2.51
            b = 0.03
            c = 2.43
            d = 0.59
            e = 0.14
            tm = np.clip((output_exposed * (a * output_exposed + b)) / 
                        (output_exposed * (c * output_exposed + d) + e), 0, 1)
        else:  # 'simple'
            tm = np.clip(output_exposed / np.percentile(output_exposed, 99.5), 0, 1)
        
        tm = np.clip(tm, 0, 1)
        
        if args.output_gamma != 1.0:
            tm = np.power(tm, 1.0 / args.output_gamma)

        cv2.imwrite(
            os.path.join(args.save_dir, f'output_{idx+1}_tm.png'),
            (tm[..., ::-1] * 255).astype(np.uint8)
        )
        
        print(f'Sample {idx+1}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f} dB')
        
        # Save debug visualization
        if args.save_debug:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            
            # Row 1: Comparison for metrics (original tone mapping)
            axes[0, 0].imshow(sr_tm_for_metrics)
            axes[0, 0].set_title(f'Output (Reinhard TM for metrics)\nSSIM={ssim_val:.4f}')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(gt_tm_for_metrics)
            axes[0, 1].set_title('Ground Truth (Reinhard TM)')
            axes[0, 1].axis('off')
            
            diff_metrics = np.abs(sr_tm_for_metrics - gt_tm_for_metrics)
            axes[0, 2].imshow(diff_metrics)
            axes[0, 2].set_title(f'Difference (for metrics)\nMean: {diff_metrics.mean():.4f}')
            axes[0, 2].axis('off')
            
            # Row 2: Visualization with exposure adjustment
            axes[1, 0].imshow(tm)
            axes[1, 0].set_title(f'Output (exposure-adjusted for display)\n{args.tonemap_method.upper()} TM')
            axes[1, 0].axis('off')
            
            # Histograms
            axes[1, 1].hist(output_np.flatten(), bins=100, color='blue', alpha=0.7, label='Output HDR')
            axes[1, 1].hist(gt_hdr.flatten(), bins=100, color='green', alpha=0.5, label='GT HDR')
            axes[1, 1].set_title('HDR Value Distribution')
            axes[1, 1].set_xlabel('HDR Value')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].set_yscale('log')
            axes[1, 1].legend()
            
            axes[1, 2].hist(sr_tm_for_metrics.flatten(), bins=100, color='blue', alpha=0.7, label='Output TM')
            axes[1, 2].hist(gt_tm_for_metrics.flatten(), bins=100, color='green', alpha=0.5, label='GT TM')
            axes[1, 2].set_title('Tone-Mapped Value Distribution')
            axes[1, 2].set_xlabel('Value')
            axes[1, 2].set_ylabel('Count')
            axes[1, 2].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, f'debug_{idx+1}.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Print statistics
            print(f'  Output HDR range: [{output_np.min():.4f}, {output_np.max():.4f}], mean: {output_np.mean():.4f}')
            print(f'  GT HDR range: [{gt_hdr.min():.4f}, {gt_hdr.max():.4f}], mean: {gt_hdr.mean():.4f}')
            print(f'  Output TM (metrics) mean: {sr_tm_for_metrics.mean():.4f}')
            print(f'  GT TM (metrics) mean: {gt_tm_for_metrics.mean():.4f}')
        
        # Clear scene data from GPU
        del t1, t2, sum1, sum2, ldr_long_t, ldr_short_t
        del ldr_long_pad, ldr_short_pad, out_acc, w_acc, out_final
        if args.device == 'cuda':
            torch.cuda.empty_cache()

    print("All scenes processed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--spa_channels', type=int, default=3)
    parser.add_argument('--spe_channels', type=int, default=3)
    parser.add_argument('--cut_size', type=int, default=64)
    parser.add_argument('--ratio', type=int, default=1)
    parser.add_argument('--pad', type=int, default=0)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--weight', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_log', action='store_true', default=True,
                        help='Apply log-space inverse transform (disable if colors look wrong)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Number of tiles to process in parallel (increase for speed, decrease for memory)')
    parser.add_argument('--use_amp', action='store_true', 
                        help='Use automatic mixed precision for faster inference')
    parser.add_argument('--tonemap_method', type=str, default='aces',
                        choices=['reinhard', 'gamma', 'log', 'simple', 'aces'],
                        help='Tone mapping method for visualization (aces recommended for natural colors)')
    parser.add_argument('--exposure', type=float, default=1.0,
                        help='Manual exposure multiplier (try 2.0-5.0 if image is dark)')
    parser.add_argument('--auto_exposure', action='store_true', default=True,
                        help='Automatically adjust exposure based on image content')
    parser.add_argument('--target_brightness', type=float, default=0.5,
                        help='Target brightness for auto-exposure (0.3-0.7 recommended)')
    parser.add_argument('--output_gamma', type=float, default=2.2,
                        help='Output gamma correction (2.2 is standard for sRGB)')
    parser.add_argument('--gamma', type=float, default=2.2,
                        help='Gamma value for gamma tone mapping method')
    parser.add_argument('--log_scale', type=float, default=1.0,
                        help='Scale factor for log tone mapping')
    parser.add_argument('--save_debug', action='store_true',
                        help='Save debug comparison images with histograms')
    parser.add_argument('--verify_summaries', action='store_true',
                        help='Verify that optimized summary computation matches original')
    args = parser.parse_args()

    test(args)