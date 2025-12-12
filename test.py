import os
import torch
import argparse
import numpy as np
import cv2
from model.u2net import U2Net as Net
from utils.tools import SSIM

def load_checkpoint_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        if 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        raise RuntimeError(f"Unexpected checkpoint format: {type(ckpt)}")
    new_state = {}
    for k, v in state_dict.items():
        new_key = k.replace('module.', '') if isinstance(k, str) else k
        new_state[new_key] = v
    return new_state

def compute_metrics(sr, gt):
    # sr and gt are torch tensors (1,C,H,W)
    from pytorch_msssim import ssim, ms_ssim

    sr = sr.clamp(0,1)
    gt = gt.clamp(0,1)

    ssim_val = ssim(sr, gt, data_range=1.0, size_average=True).item()
    mse = ((sr - gt) ** 2).mean()
    psnr_val = 10 * torch.log10(1.0 / mse).item()
    return ssim_val, psnr_val


def test(args):
    model = Net(args.channels, args.spa_channels, args.spe_channels,
                args.cut_size + 2*args.pad, args.cut_size + 2*args.pad, args.ratio).to(args.device)
    state_dict = load_checkpoint_state_dict(args.weight, args.device)
    model.load_state_dict(state_dict)
    model.eval()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Scan test folder for scenes
    scenes = sorted([d for d in os.listdir(args.test_data_path) if os.path.isdir(os.path.join(args.test_data_path, d))])
    for idx, scene in enumerate(scenes):
        scene_path = os.path.join(args.test_data_path, scene)
        ldr_files = sorted([f for f in os.listdir(scene_path) if f.startswith("input_") and f.endswith("_aligned.tif")])
        hdr_files = sorted([f for f in os.listdir(scene_path) if f.startswith("ref_hdr") and f.endswith("_aligned.hdr")])
        if len(hdr_files) == 0 or len(ldr_files) < 2:
            continue

        ldr_long = cv2.imread(os.path.join(scene_path, ldr_files[0]), cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
        ldr_short = cv2.imread(os.path.join(scene_path, ldr_files[1]), cv2.IMREAD_UNCHANGED).astype(np.float32)/255.0
        gt_hdr = cv2.imread(os.path.join(scene_path, hdr_files[0]), -1).astype(np.float32)

        # Convert BGR->RGB
        if ldr_long.ndim == 3 and ldr_long.shape[2] == 3:
            ldr_long = cv2.cvtColor(ldr_long, cv2.COLOR_BGR2RGB)
            ldr_short = cv2.cvtColor(ldr_short, cv2.COLOR_BGR2RGB)
        if gt_hdr.ndim == 3 and gt_hdr.shape[2] == 3:
            gt_hdr = cv2.cvtColor(gt_hdr, cv2.COLOR_BGR2RGB)

        # Optional log
        use_log = getattr(args, 'use_log', True)
        if use_log:
            gt_c = torch.from_numpy(np.log1p(gt_hdr)).permute(2,0,1).unsqueeze(0).float().to(args.device)
        else:
            gt_c = torch.from_numpy(gt_hdr).permute(2,0,1).unsqueeze(0).float().to(args.device)

        # Patch-based inference
        h, w, C = ldr_long.shape
        pad = args.pad
        cut_size = args.cut_size
        ms_size = cut_size // args.ratio
        edge_H = (cut_size - (h % cut_size)) % cut_size
        edge_W = (cut_size - (w % cut_size)) % cut_size

        ldr_long_t = torch.from_numpy(ldr_long).permute(2,0,1).unsqueeze(0).float().to(args.device)
        ldr_short_t = torch.from_numpy(ldr_short).permute(2,0,1).unsqueeze(0).float().to(args.device)

        # Pad
        ldr_long_pad = torch.nn.functional.pad(ldr_long_t, (pad, pad, pad, pad), 'reflect')
        ldr_short_pad = torch.nn.functional.pad(ldr_short_t, (pad//args.ratio, pad//args.ratio, pad//args.ratio, pad//args.ratio), 'reflect')

        ldr_long_pad = torch.nn.functional.pad(ldr_long_pad, (0, edge_W, 0, edge_H), 'reflect')
        ldr_short_pad = torch.nn.functional.pad(ldr_short_pad, (0, edge_W//args.ratio, 0, edge_H//args.ratio), 'reflect')

        H_pad, W_pad = ldr_long_pad.shape[2], ldr_long_pad.shape[3]
        output = torch.zeros(1, C, H_pad, W_pad).to(args.device)

        scale_H = H_pad // cut_size
        scale_W = W_pad // cut_size

        for i in range(scale_H):
            for j in range(scale_W):
                MS_patch = ldr_short_pad[:, :, i*ms_size:i*ms_size+ms_size+pad//2, j*ms_size:j*ms_size+ms_size+pad//2]
                PAN_patch = ldr_long_pad[:, :, i*cut_size:i*cut_size+cut_size+2*pad, j*cut_size:j*cut_size+cut_size+2*pad]
                with torch.no_grad():
                    sr_patch = model(MS_patch, PAN_patch)
                    sr_patch = torch.clamp(sr_patch, 0, 1)
                output[:, :, i*cut_size:(i+1)*cut_size, j*cut_size:(j+1)*cut_size] = \
                    sr_patch[:, :, pad:cut_size+pad, pad:cut_size+pad]

        output = output[:, :, :h, :w].squeeze().permute(1,2,0).cpu().numpy()

        # --- Convert back from log domain if needed ---
        if use_log:
            output_lin = np.expm1(output)
        else:
            output_lin = output
        output_lin = np.clip(output_lin, 0.0, None)

        # Save HDR
        save_path_hdr = os.path.join(args.save_dir, f'output_{idx+1}.hdr')
        cv2.imwrite(save_path_hdr, output_lin[..., ::-1].astype(np.float32))  # RGB->BGR for OpenCV
        print(f'Saved HDR: {save_path_hdr}')

        # Tone-map for visualization / SSIM/PSNR
        tm = output_lin / (1.0 + output_lin)
        tm = np.clip(tm, 0.0, 1.0)
        save_path_tm = os.path.join(args.save_dir, f'output_{idx+1}_tm.png')
        cv2.imwrite(save_path_tm, (tm[..., ::-1]*255.0).astype(np.uint8))
        print(f'Saved tone-mapped PNG: {save_path_tm}')

        # Compute metrics on tone-mapped images
        sr_tm_t = torch.from_numpy(tm).permute(2,0,1).unsqueeze(0).float().to(args.device)
        gt_tm = gt_hdr / (1.0 + gt_hdr)
        gt_tm = np.clip(gt_tm, 0.0, 1.0)
        gt_tm_t = torch.from_numpy(gt_tm).permute(2,0,1).unsqueeze(0).float().to(args.device)

        ssim_val, psnr_val = compute_metrics(sr_tm_t, gt_tm_t)
        print(f'Sample {idx+1}: SSIM={ssim_val:.4f}, PSNR={psnr_val:.2f} dB\n')

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
    args = parser.parse_args()

    test(args)
