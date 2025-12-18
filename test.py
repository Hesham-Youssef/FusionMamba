#!/usr/bin/env python3
# tester_tiled_optimized_FIXED.py
"""
FIXED version with correct HDR handling and optimized tiling:
1. Use imageio for HDR loading (matches training)
2. Remove premature clamping of model output
3. Apply proper inverse transformation (expm1)
4. Use optimized tiling functions from utils.tools
5. CRITICAL: Use unified image_utils to ensure exact consistency with training
   - Same tensor conversion (_to_tensor_fast)
   - Same image loading (load_image_fast)
   - Same preprocessing (prepare_tensors_for_inference)
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
import imageio.v2 as imageio

# Import optimized tiling functions
from utils.tools import compute_summary_fast, extract_tiles_optimized, make_hann_window, load_image_fast, prepare_tensors_for_inference

def test(args):
    # Model load
    from model.u2net import U2Net as Net
    model = Net(
        args.channels,
        args.spa_channels,
        args.spe_channels,
        args.cut_size + 2 * args.pad,
        args.cut_size + 2 * args.pad
    ).to(args.device)

    state_dict = torch.load(args.weight, map_location=args.device)    
    if isinstance(state_dict, dict):
        if 'model_state' in state_dict:
            state_dict = state_dict['model_state']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    
    # Enable mixed precision if requested
    use_amp = args.use_amp and args.device == 'cuda'
    if use_amp:
        print("Using automatic mixed precision (AMP) for model inference")
        print("Note: Summaries always computed in float32 for consistency with training")
    
    if args.batch_size > 1:
        print(f"Using batch processing with batch_size={args.batch_size}")
    else:
        print("Using single-tile processing")

    os.makedirs(args.save_dir, exist_ok=True)

    scenes = sorted(
        d for d in os.listdir(args.test_data_path)
        if os.path.isdir(os.path.join(args.test_data_path, d))
    )

    import glob
    for idx, scene in enumerate(scenes):
        scene_path = os.path.join(args.test_data_path, scene)

        ldr_files = sorted(
            f for f in glob.glob(os.path.join(scene_path, "input_*.tif"))
            if "_aligned" not in os.path.basename(f)
        )
        hdr_files = glob.glob(os.path.join(scene_path, "ref_hdr.hdr"))
        if len(ldr_files) < 3 or len(hdr_files) == 0:
            print(f"Skipping scene {scene}: insufficient files")
            continue

        print(f"\n{'='*80}")
        print(f"Processing Scene {idx+1}: {scene}")
        print(f"{'='*80}")

        # Read all 3 LDR images and ground truth
        img0 = load_image_fast(ldr_files[0])  # Darkest exposure
        img1 = load_image_fast(ldr_files[1])  # Middle exposure
        img2 = load_image_fast(ldr_files[2])  # Brightest exposure
        gt_hdr = load_image_fast(hdr_files[0])  # Ground truth HDR

        # Process both pairs
        pairs = [
            ("pair_1-0", img1, img0, "middle+dark"),
            ("pair_1-2", img1, img2, "middle+bright")
        ]
        
        hdr_results = {}
        
        for pair_name, img_a, img_b, pair_desc in pairs:
            print(f"\n--- Processing {pair_desc} ({pair_name}) ---")
            
            # Convert to tensors
            t_a, t_b = prepare_tensors_for_inference(img_a, img_b, device=args.device)
            
            H, W = t_a.shape[1], t_a.shape[2]

            # Compute summaries
            cut = args.cut_size
            stride = cut // 2

            tiles_a, coords_a, orig_shape_a = extract_tiles_optimized(
                t_a, cut, cut, stride, stride, pad_mode='reflect'
            )
            tiles_b, coords_b, orig_shape_b = extract_tiles_optimized(
                t_b, cut, cut, stride, stride, pad_mode='reflect'
            )

            sum_a = compute_summary_fast(t_a, tiles_a, coords_a, orig_shape_a, cut, cut)
            sum_b = compute_summary_fast(t_b, tiles_b, coords_b, orig_shape_b, cut, cut)
            
            if args.save_debug:
                print(f'  sum_a: min={sum_a.min():.4f}, max={sum_a.max():.4f}, mean={sum_a.mean():.4f}')
                print(f'  sum_b: min={sum_b.min():.4f}, max={sum_b.max():.4f}, mean={sum_b.mean():.4f}')
            
            # Prepare for tiled inference
            pad = args.pad
            ratio = args.ratio
            ms_size = cut // ratio
            
            img_a_t = t_a.unsqueeze(0)
            img_b_t = t_b.unsqueeze(0)

            img_a_pad = F.pad(img_a_t, (pad, pad, pad, pad), 'reflect')
            img_b_pad = F.pad(
                img_b_t,
                (pad // ratio, pad // ratio, pad // ratio, pad // ratio),
                'reflect'
            )

            edge_H = (cut - (H % cut)) % cut
            edge_W = (cut - (W % cut)) % cut

            img_a_pad = F.pad(img_a_pad, (0, edge_W, 0, edge_H), 'reflect')
            img_b_pad = F.pad(
                img_b_pad,
                (0, edge_W // ratio, 0, edge_H // ratio),
                'reflect'
            )

            H_pad, W_pad = img_a_pad.shape[2:]
            C = img_a_pad.shape[1]

            # Accumulators
            out_acc = torch.zeros((1, C, H_pad, W_pad), device=args.device)
            w_acc = torch.zeros((1, 1, H_pad, W_pad), device=args.device)

            win = make_hann_window(cut, cut, device_str=str(args.device)).unsqueeze(0)

            sum_a_b = sum_a.unsqueeze(0)
            sum_b_b = sum_b.unsqueeze(0)

            # Collect tile positions
            tile_positions = []
            for y in range(0, H_pad - cut + 1, stride):
                for x in range(0, W_pad - cut + 1, stride):
                    tile_positions.append((y, x))

            batch_size = args.batch_size

            with torch.no_grad():
                autocast_context = torch.cuda.amp.autocast() if use_amp else nullcontext()
                
                for batch_start in range(0, len(tile_positions), batch_size):
                    batch_end = min(batch_start + batch_size, len(tile_positions))
                    batch_tiles = tile_positions[batch_start:batch_end]
                    
                    img_a_batch = []
                    img_b_batch = []
                    
                    for y, x in batch_tiles:
                        img_b_patch = img_b_pad[
                            :, :,
                            y // ratio : y // ratio + ms_size + pad // 2,
                            x // ratio : x // ratio + ms_size + pad // 2
                        ]
                        img_a_patch = img_a_pad[
                            :, :,
                            y : y + cut + 2 * pad,
                            x : x + cut + 2 * pad
                        ]
                        img_a_batch.append(img_a_patch)
                        img_b_batch.append(img_b_patch)
                    
                    img_a_batch = torch.cat(img_a_batch, dim=0)
                    img_b_batch = torch.cat(img_b_batch, dim=0)
                    
                    sum_a_batch = sum_a_b.expand(len(batch_tiles), -1, -1, -1)
                    sum_b_batch = sum_b_b.expand(len(batch_tiles), -1, -1, -1)
                    
                    with autocast_context:
                        sr_batch, _, _, _ = model(img_a_batch, img_b_batch, sum_a_batch, sum_b_batch)
                    
                    sr_batch = sr_batch.float()
                    sr_batch = torch.expm1(sr_batch)
                    
                    for i, (y, x) in enumerate(batch_tiles):
                        sr_center = sr_batch[i:i+1, :, pad:pad + cut, pad:pad + cut]
                        out_acc[:, :, y:y + cut, x:x + cut] += sr_center * win
                        w_acc[:, :, y:y + cut, x:x + cut] += win
                    
                    del img_a_batch, img_b_batch, sum_a_batch, sum_b_batch, sr_batch
                    if args.device == 'cuda':
                        torch.cuda.empty_cache()

            out_final = out_acc / (w_acc + 1e-8)
            out_final = out_final[:, :, :H, :W]
            
            # Store result
            hdr_results[pair_name] = out_final.squeeze(0).permute(1, 2, 0).cpu().numpy()
            
            print(f"  {pair_name} HDR range: [{hdr_results[pair_name].min():.4f}, {hdr_results[pair_name].max():.4f}]")
            print(f"  {pair_name} HDR mean: {hdr_results[pair_name].mean():.4f}")

        # Save individual results
        print(f"\nSaving outputs for scene {scene}...")
        
        for pair_name, hdr_output in hdr_results.items():
            output_path = os.path.join(args.save_dir, f"{scene}_{pair_name}.hdr")
            cv2.imwrite(output_path, hdr_output[:, :, ::-1].astype(np.float32))
            print(f"  Saved: {output_path}")
        
        # Optional: Merge the two HDR results
        if args.merge_pairs:
            print(f"\nMerging pairs for scene {scene}...")
            hdr_merged = merge_hdr_pairs(
                hdr_results["pair_1-0"], 
                hdr_results["pair_1-2"],
                method=args.merge_method  # 'average', 'max', 'weighted'
            )
            merged_path = os.path.join(args.save_dir, f"{scene}_merged.hdr")
            cv2.imwrite(merged_path, hdr_merged[:, :, ::-1].astype(np.float32))
            print(f"  Saved merged: {merged_path}")
            
            # Compute metrics on merged result
            if args.compute_metrics:
                gt_np = gt_hdr if isinstance(gt_hdr, np.ndarray) else gt_hdr.cpu().numpy()
                compute_and_print_metrics(hdr_merged, gt_np, f"Scene {scene} (merged)")
        
        # Compute metrics for individual pairs
        if args.compute_metrics:
            gt_np = gt_hdr if isinstance(gt_hdr, np.ndarray) else gt_hdr.cpu().numpy()
            for pair_name, hdr_output in hdr_results.items():
                compute_and_print_metrics(hdr_output, gt_np, f"Scene {scene} ({pair_name})")

    print("\n" + "="*80)
    print("All scenes processed.")
    print("="*80)


def merge_hdr_pairs(hdr1, hdr2, method='weighted'):
    """
    Merge two HDR images from different exposure pairs.
    
    Args:
        hdr1: HDR from (middle, dark) pair - better for bright regions
        hdr2: HDR from (middle, bright) pair - better for dark regions
        method: 'average', 'max', 'weighted', or 'adaptive'
    """
    if method == 'average':
        return (hdr1 + hdr2) / 2.0
    
    elif method == 'max':
        # Take maximum value at each pixel (preserves brightest)
        return np.maximum(hdr1, hdr2)
    
    elif method == 'weighted':
        # Weight based on intensity - dark pair for highlights, bright pair for shadows
        luminance1 = 0.2126 * hdr1[..., 0] + 0.7152 * hdr1[..., 1] + 0.0722 * hdr1[..., 2]
        luminance2 = 0.2126 * hdr2[..., 0] + 0.7152 * hdr2[..., 1] + 0.0722 * hdr2[..., 2]
        
        # Create weights: use hdr1 (with dark) for bright regions, hdr2 (with bright) for dark regions
        # Normalize luminances
        lum1_norm = luminance1 / (luminance1 + luminance2 + 1e-6)
        lum2_norm = luminance2 / (luminance1 + luminance2 + 1e-6)
        
        # Apply weights
        weight1 = lum1_norm[..., None]
        weight2 = lum2_norm[..., None]
        
        return hdr1 * weight1 + hdr2 * weight2
    
    elif method == 'adaptive':
        # Adaptive merging: use confidence based on which pair should perform better
        # hdr1 (middle+dark) is better for bright regions
        # hdr2 (middle+bright) is better for dark regions
        
        intensity_threshold = 128.0  # Crossover point
        
        # Compute average intensity
        avg_intensity1 = (hdr1[..., 0] + hdr1[..., 1] + hdr1[..., 2]) / 3.0
        avg_intensity2 = (hdr2[..., 0] + hdr2[..., 1] + hdr2[..., 2]) / 3.0
        
        # Create smooth transition weights
        # Sigmoid function for smooth blending
        x = (avg_intensity1 + avg_intensity2) / 2.0
        weight1 = 1.0 / (1.0 + np.exp(-0.05 * (x - intensity_threshold)))
        weight2 = 1.0 - weight1
        
        weight1 = weight1[..., None]
        weight2 = weight2[..., None]
        
        return hdr1 * weight1 + hdr2 * weight2
    
    else:
        raise ValueError(f"Unknown merge method: {method}")


def compute_and_print_metrics(pred_hdr, gt_hdr, label=""):
    """Compute and print PSNR and SSIM metrics."""
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    
    # Ensure same shape
    if pred_hdr.shape != gt_hdr.shape:
        print(f"  Warning: Shape mismatch for {label}")
        return
    
    # Clip predictions to valid range
    pred_hdr_clipped = np.clip(pred_hdr, 0, 255)
    
    # Compute metrics
    psnr_val = psnr(gt_hdr, pred_hdr_clipped, data_range=255.0)
    ssim_val = ssim(gt_hdr, pred_hdr_clipped, data_range=255.0, channel_axis=2)
    
    print(f"\n  [{label}]")
    print(f"  PSNR: {psnr_val:.2f} dB")
    print(f"  SSIM: {ssim_val:.4f}")
    print(f"  Output range: [{pred_hdr.min():.2f}, {pred_hdr.max():.2f}]")
    print(f"  GT range: [{gt_hdr.min():.2f}, {gt_hdr.max():.2f}]")

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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--use_amp', action='store_true')
    parser.add_argument('--tonemap_method', type=str, default='aces',
                        choices=['reinhard', 'gamma', 'log', 'simple', 'aces'])
    parser.add_argument('--exposure', type=float, default=1.0)
    parser.add_argument('--auto_exposure', action='store_true', default=True)
    parser.add_argument('--target_brightness', type=float, default=0.5)
    parser.add_argument('--output_gamma', type=float, default=2.2)
    parser.add_argument('--gamma', type=float, default=2.2)
    parser.add_argument('--log_scale', type=float, default=1.0)
    parser.add_argument('--save_debug', action='store_true')
    parser.add_argument('--merge_pairs', action='store_true', 
                    help='Merge the two HDR pairs into a single result')

    parser.add_argument('--merge_method', type=str, default='adaptive',
                        choices=['average', 'max', 'weighted', 'adaptive'],
                        help='Method for merging HDR pairs: '
                            'average (simple mean), '
                            'max (take brightest), '
                            'weighted (luminance-based), '
                            'adaptive (intensity-based sigmoid)')

    parser.add_argument('--compute_metrics', action='store_true',
                        help='Compute PSNR and SSIM metrics against ground truth')
    args = parser.parse_args()

    test(args)