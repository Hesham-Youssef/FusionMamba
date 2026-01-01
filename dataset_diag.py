import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils.tools import *


def comprehensive_diagnostics(pred, target, img1, img2, sum1, sum2, name="batch"):
    """Comprehensive diagnostics for debugging HDR reconstruction"""
    
    with torch.no_grad():
        print(f"\n{'='*80}")
        print(f"DIAGNOSTICS: {name}")
        print(f"{'='*80}")
        
        # Input statistics
        print("\n📊 INPUT STATISTICS:")
        print(f"  img1 (LDR+HDR): [{img1.min():.3f}, {img1.max():.3f}] μ={img1.mean():.3f} σ={img1.std():.3f}")
        print(f"  img2 (LDR+HDR): [{img2.min():.3f}, {img2.max():.3f}] μ={img2.mean():.3f} σ={img2.std():.3f}")
        print(f"  sum1 (global): [{sum1.min():.3f}, {sum1.max():.3f}] μ={sum1.mean():.3f} σ={sum1.std():.3f}")
        print(f"  sum2 (global): [{sum2.min():.3f}, {sum2.max():.3f}] μ={sum2.mean():.3f} σ={sum2.std():.3f}")
        
        # Output vs Target
        print("\n🎯 OUTPUT vs TARGET:")
        print(f"  Prediction: [{pred.min():.3f}, {pred.max():.3f}] μ={pred.mean():.3f} σ={pred.std():.3f}")
        print(f"  Target:     [{target.min():.3f}, {target.max():.3f}] μ={target.mean():.3f} σ={target.std():.3f}")
        
        # Check if output is collapsed
        pred_range = pred.max() - pred.min()
        target_range = target.max() - target.min()
        print(f"  Dynamic Range Ratio: {pred_range/target_range:.4f} (should be close to 1.0)")
        
        # FIXED: Data is already log-compressed, just map to [0,1] for display
        pred_display = (pred + 1) / 2
        target_display = (target + 1) / 2
        print("\n🎨 DISPLAY RANGE (already log-compressed):")
        print(f"  Pred [0,1]: [{pred_display.min():.3f}, {pred_display.max():.3f}] μ={pred_display.mean():.3f}")
        print(f"  Targ [0,1]: [{target_display.min():.3f}, {target_display.max():.3f}] μ={target_display.mean():.3f}")
        
        # Check for NaN/Inf
        if torch.isnan(pred).any():
            print("  ⚠️  WARNING: NaN detected in prediction!")
        if torch.isinf(pred).any():
            print("  ⚠️  WARNING: Inf detected in prediction!")
        
        # Histogram analysis
        pred_flat = pred.flatten().cpu().numpy()
        target_flat = target.flatten().cpu().numpy()
        
        print("\n📈 HISTOGRAM ANALYSIS:")
        for percentile in [1, 5, 25, 50, 75, 95, 99]:
            pred_p = np.percentile(pred_flat, percentile)
            targ_p = np.percentile(target_flat, percentile)
            print(f"  P{percentile:2d}: pred={pred_p:6.3f}, target={targ_p:6.3f}")
        
        # Per-channel analysis
        print("\n🌈 PER-CHANNEL ANALYSIS:")
        for c, color in enumerate(['R', 'G', 'B']):
            pred_c = pred[:, c, :, :]
            targ_c = target[:, c, :, :]
            print(f"  {color}: pred=[{pred_c.min():.3f}, {pred_c.max():.3f}] "
                  f"target=[{targ_c.min():.3f}, {targ_c.max():.3f}]")
        
        # Exposure check (ratio of bright to dark regions)
        pred_bright = (pred > 0.5).sum().item()
        pred_dark = (pred < -0.5).sum().item()
        targ_bright = (target > 0.5).sum().item()
        targ_dark = (target < -0.5).sum().item()
        
        print("\n💡 EXPOSURE DISTRIBUTION:")
        print(f"  Prediction: {pred_bright} bright, {pred_dark} dark pixels")
        print(f"  Target:     {targ_bright} bright, {targ_dark} dark pixels")
        
        
        psnr_log, ssim_log = compute_metrics(pred, target, use_hdr=False)
        psnr_hdr, ssim_hdr = compute_metrics(pred, target, use_hdr=True)

        print("\n📐 QUALITY METRICS:")
        print(f"  Log-compressed space:")
        print(f"    PSNR: {psnr_log:.2f} dB")
        print(f"    SSIM: {ssim_log:.4f}")

        print(f"  Linear HDR space:")
        print(f"    PSNR: {psnr_hdr:.2f} dB")
        print(f"    SSIM: {ssim_hdr:.4f}")
        
        print(f"{'='*80}\n")


def save_diagnostic_images(pred, target, img1, img2, sample_path, idx=0):
    """Save detailed diagnostic images with PSNR & SSIM"""
    
    os.makedirs(sample_path, exist_ok=True)
    
    with torch.no_grad():
        # -----------------------------
        # Display conversion
        # -----------------------------
        def to_displayable(tensor):
            out = (tensor + 1) / 2  # [-1,1] → [0,1]
            out = torch.clamp(out, 0, 1)
            out = out[0].cpu().numpy().transpose(1, 2, 0)
            return (out * 255).astype(np.uint8)

        pred_img   = to_displayable(pred)
        target_img = to_displayable(target)
        img1_img   = to_displayable(img1)
        img2_img   = to_displayable(img2)

        # -----------------------------
        # PSNR & SSIM (linear)
        # -----------------------------
        psnr_val, ssim_val = compute_metrics(pred, target, use_hdr=True)


        # -----------------------------
        # Error maps
        # -----------------------------
        error = torch.abs(pred - target)
        error_max = error.max()
        if error_max > 0:
            error = error / error_max

        error_img = (error[0].mean(dim=0).cpu().numpy() * 255).astype(np.uint8)
        error_img = cv2.applyColorMap(error_img, cv2.COLORMAP_JET)

        # -----------------------------
        # Composite
        # -----------------------------
        h, w = pred_img.shape[:2]
        composite = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)

        # Top row
        composite[0:h, 0:w]       = img1_img
        composite[0:h, w:2*w]     = img2_img
        composite[0:h, 2*w:3*w]   = pred_img

        # Bottom row
        composite[h:2*h, 0:w]     = target_img
        composite[h:2*h, w:2*w]   = error_img
        composite[h:2*h, 2*w:3*w] = np.abs(
            pred_img.astype(np.int16) - target_img.astype(np.int16)
        ).astype(np.uint8)

        # -----------------------------
        # Labels
        # -----------------------------
        labels = [
            ("Input 1", (10, 30)),
            ("Input 2", (w + 10, 30)),
            ("Output", (2*w + 10, 30)),
            ("Target", (10, h + 30)),
            ("Error (heat)", (w + 10, h + 30)),
            ("Error (diff)", (2*w + 10, h + 30)),
        ]

        for text, pos in labels:
            cv2.putText(
                composite, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2
            )

        metrics_text = [
            f"PSNR: {psnr_val:.2f} dB",
            f"SSIM: {ssim_val:.4f}",
        ]

        y0 = 60
        for i, text in enumerate(metrics_text):
            cv2.putText(
                composite,
                text,
                (2*w + 10, y0 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
        save_path = os.path.join(sample_path, f"diagnostic_{idx:04d}.png")
        cv2.imwrite(save_path, composite[..., ::-1])  # RGB → BGR
        print(f"    💾 Saved diagnostic: {save_path}")
