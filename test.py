#!/usr/bin/env python3
# tester_tiled.py
"""
Patch-based tiled tester that:
 - tiles the two full-resolution LDR images,
 - computes pair summaries (sum1, sum2) using Hann-window merging + adaptive pooling
 - runs model inference per tile (keeps same forward signature as your original tester)
 - merges tiles back to full-resolution using Hann-window weights
 - saves HDR and tone-mapped PNG and computes SSIM/PSNR on tone-mapped images
"""

import os
import argparse
from pathlib import Path

import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F

# --- copy/short versions of the utilities used in your loader ---
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

def extract_tiles(img, tile_h, tile_w, stride_h=None, stride_w=None, pad_mode='reflect'):
    """Extract tiles from a torch tensor `img` shaped (C,H,W).
       Returns: tiles (list of tensors), coords list of (y0,y1,x0,x1), orig_shape (H,W), pads tuple
    """
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 3
    C, H, W = img.shape
    stride_h = stride_h or tile_h
    stride_w = stride_w or tile_w

    # compute padding so tiles fully cover the image
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


def _compute_summary(ldr1, ldr2, tiles1, tiles2, coords, orig_shape, tile_h, tile_w):
    """Merge tile lists back to full image using Hann weighting then adaptive-pool to (tile_h,tile_w).
       Returns sum1, sum2 tensors of shape (C, tile_h, tile_w) - same as loader.
    """
    H, W = orig_shape
    device = ldr1.device
    dtype = ldr1.dtype
    win = make_hann_window(tile_h, tile_w, device=device, dtype=dtype)

    def merge(tiles, base):
        out = torch.zeros_like(base)
        wgt = torch.zeros((1, H, W), dtype=base.dtype, device=device)
        for tile, (y0, y1, x0, x1) in zip(tiles, coords):
            vh, vw = y1 - y0, x1 - x0
            ws = win[:vh, :vw].unsqueeze(0)  # (1, vh, vw)
            out[:, y0:y1, x0:x1] += tile[:, :vh, :vw] * ws
            wgt[:, y0:y1, x0:x1] += ws
        return out / (wgt + 1e-8)

    s1 = merge(tiles1, ldr1)
    s2 = merge(tiles2, ldr2)

    # downsample merged full image to (tile_h, tile_w) summary
    s1_pool = F.adaptive_avg_pool2d(s1.unsqueeze(0), (tile_h, tile_w)).squeeze(0)
    s2_pool = F.adaptive_avg_pool2d(s2.unsqueeze(0), (tile_h, tile_w)).squeeze(0)
    return s1_pool, s2_pool


# --- metrics helper (same as your tester) ---
def compute_metrics(sr, gt):
    from pytorch_msssim import ssim
    sr = sr.clamp(0,1)
    gt = gt.clamp(0,1)
    ssim_val = ssim(sr, gt, data_range=1.0, size_average=True).item()
    mse = ((sr - gt) ** 2).mean()
    psnr_val = 10 * torch.log10(1.0 / (mse + 1e-12)).item()
    return ssim_val, psnr_val

def test(args):
    # ------------------------------------------------------------
    # Model load
    # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Read images
        # ------------------------------------------------------------
        ldr_long = cv2.imread(os.path.join(scene_path, ldr_files[0]), -1).astype(np.float32) / 255.0
        ldr_short = cv2.imread(os.path.join(scene_path, ldr_files[1]), -1).astype(np.float32) / 255.0
        gt_hdr = cv2.imread(os.path.join(scene_path, hdr_files[0]), -1).astype(np.float32)

        if ldr_long.ndim == 3:
            ldr_long = cv2.cvtColor(ldr_long, cv2.COLOR_BGR2RGB)
            ldr_short = cv2.cvtColor(ldr_short, cv2.COLOR_BGR2RGB)
        if gt_hdr.ndim == 3:
            gt_hdr = cv2.cvtColor(gt_hdr, cv2.COLOR_BGR2RGB)

        use_log = args.use_log

        # ------------------------------------------------------------
        # To tensors
        # ------------------------------------------------------------
        t1 = _to_tensor_and_normalize(ldr_long).unsqueeze(0).to(args.device)
        t2 = _to_tensor_and_normalize(ldr_short).unsqueeze(0).to(args.device)

        H, W = t1.shape[2], t1.shape[3]

        # ------------------------------------------------------------
        # Compute global summaries (once per scene)
        # ------------------------------------------------------------
        cut = args.cut_size
        stride = cut // 2

        tiles1, coords, orig_shape, _ = extract_tiles(
            t1.squeeze(0), cut, cut, stride, stride
        )
        tiles2, _, _, _ = extract_tiles(
            t2.squeeze(0), cut, cut, stride, stride
        )

        sum1, sum2 = _compute_summary(
            t1.squeeze(0), t2.squeeze(0),
            tiles1, tiles2, coords, orig_shape,
            cut, cut
        )

        # ------------------------------------------------------------
        # Padding for tiled inference
        # ------------------------------------------------------------
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

        # ------------------------------------------------------------
        # Accumulators
        # ------------------------------------------------------------
        out_acc = torch.zeros((1, C, H_pad, W_pad), device=args.device)
        w_acc = torch.zeros((1, 1, H_pad, W_pad), device=args.device)

        win = make_hann_window(cut, cut, device=args.device).unsqueeze(0)

        sum1_b = sum1.unsqueeze(0)
        sum2_b = sum2.unsqueeze(0)

        # ------------------------------------------------------------
        # CLEAN tiled inference loop (NO SEAMS)
        # ------------------------------------------------------------
        with torch.no_grad():
            for y in range(0, H_pad - cut + 1, stride):
                for x in range(0, W_pad - cut + 1, stride):

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

                    sr = model(MS_patch, PAN_patch, sum1_b, sum2_b).clamp(0, 1)

                    sr_center = sr[:, :, pad:pad + cut, pad:pad + cut]

                    out_acc[:, :, y:y + cut, x:x + cut] += sr_center * win
                    w_acc[:, :, y:y + cut, x:x + cut] += win

        out_final = out_acc / (w_acc + 1e-8)
        out_final = out_final[:, :, :H, :W]

        # ------------------------------------------------------------
        # Save outputs
        # ------------------------------------------------------------
        output_np = out_final.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if use_log:
            output_np = np.expm1(output_np)
        output_np = np.clip(output_np, 0, None)

        cv2.imwrite(
            os.path.join(args.save_dir, f'output_{idx+1}.hdr'),
            output_np[..., ::-1].astype(np.float32)
        )

        tm = output_np / (1.0 + output_np)
        tm = np.clip(tm, 0, 1)

        cv2.imwrite(
            os.path.join(args.save_dir, f'output_{idx+1}_tm.png'),
            (tm[..., ::-1] * 255).astype(np.uint8)
        )

        sr_tm = torch.from_numpy(tm).permute(2, 0, 1).unsqueeze(0).to(args.device)
        gt_tm = torch.from_numpy(gt_hdr / (1 + gt_hdr)).permute(2, 0, 1).unsqueeze(0).to(args.device)

        ssim_val, psnr_val = compute_metrics(sr_tm, gt_tm)
        print(f'Sample {idx+1}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f} dB')

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
    parser.add_argument('--use_log', action='store_true', default=True)
    args = parser.parse_args()

    test(args)
