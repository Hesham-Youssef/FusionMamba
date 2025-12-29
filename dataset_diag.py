import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from utils.tools import *


def tonemap(hdr_image, mu=5000.0, gamma=2.2):
    tonemapped = np.log1p(mu * hdr_image) / np.log1p(mu)
    tonemapped = np.power(np.clip(tonemapped, 0, 1), 1.0 / gamma)
    return tonemapped


def comprehensive_diagnostics(pred, target, img1, img2, sum1, sum2, name="batch"):
    """Comprehensive diagnostics for debugging HDR reconstruction"""
    
    with torch.no_grad():
        print(f"\n{'='*80}")
        print(f"DIAGNOSTICS: {name}")
        print(f"{'='*80}")
        
        # Input statistics (in compressed [-1, 1] space)
        print("\n📊 INPUT STATISTICS (compressed space [-1, 1]):")
        print(f"  img1 (LDR+HDR): [{img1.min():.3f}, {img1.max():.3f}] μ={img1.mean():.3f} σ={img1.std():.3f}")
        print(f"  img2 (LDR+HDR): [{img2.min():.3f}, {img2.max():.3f}] μ={img2.mean():.3f} σ={img2.std():.3f}")
        print(f"  sum1 (global): [{sum1.min():.3f}, {sum1.max():.3f}] μ={sum1.mean():.3f} σ={sum1.std():.3f}")
        print(f"  sum2 (global): [{sum2.min():.3f}, {sum2.max():.3f}] μ={sum2.mean():.3f} σ={sum2.std():.3f}")
        
        # Convert to HDR radiance space for meaningful analysis
        pred_hdr = inverse_transform(pred)
        target_hdr = inverse_transform(target)
        
        # Output vs Target (compressed space)
        print("\n🎯 OUTPUT vs TARGET (compressed space [-1, 1]):")
        print(f"  Prediction: [{pred.min():.3f}, {pred.max():.3f}] μ={pred.mean():.3f} σ={pred.std():.3f}")
        print(f"  Target:     [{target.min():.3f}, {target.max():.3f}] μ={target.mean():.3f} σ={target.std():.3f}")
        
        # Dynamic range in compressed space
        pred_range = pred.max() - pred.min()
        target_range = target.max() - target.min()
        print(f"  Compressed Range Ratio: {pred_range/target_range:.4f} (should be ~1.0)")
        
        # HDR radiance space statistics
        print("\n🌟 HDR RADIANCE SPACE:")
        print(f"  Prediction: [{pred_hdr.min():.6f}, {pred_hdr.max():.3f}] μ={pred_hdr.mean():.3f}")
        print(f"  Target:     [{target_hdr.min():.6f}, {target_hdr.max():.3f}] μ={target_hdr.mean():.3f}")
        
        # HDR dynamic range (more meaningful than compressed)
        pred_hdr_range = pred_hdr.max() / (pred_hdr.min() + 1e-8)
        target_hdr_range = target_hdr.max() / (target_hdr.min() + 1e-8)
        print(f"  HDR Dynamic Range: pred={pred_hdr_range:.1f}:1, target={target_hdr_range:.1f}:1")
        
        # Check for NaN/Inf
        if torch.isnan(pred).any():
            print("  ⚠️  WARNING: NaN detected in prediction!")
        if torch.isinf(pred).any():
            print("  ⚠️  WARNING: Inf detected in prediction!")
        
        # Histogram analysis (in compressed space)
        pred_flat = pred.flatten().cpu().numpy()
        target_flat = target.flatten().cpu().numpy()
        
        print("\n📈 HISTOGRAM ANALYSIS (compressed space):")
        for percentile in [1, 5, 25, 50, 75, 95, 99]:
            pred_p = np.percentile(pred_flat, percentile)
            targ_p = np.percentile(target_flat, percentile)
            print(f"  P{percentile:2d}: pred={pred_p:6.3f}, target={targ_p:6.3f}")
        
        # HDR histogram analysis
        pred_hdr_flat = pred_hdr.flatten().cpu().numpy()
        target_hdr_flat = target_hdr.flatten().cpu().numpy()
        
        print("\n📊 HDR RADIANCE HISTOGRAM:")
        for percentile in [50, 75, 90, 95, 99]:
            pred_p = np.percentile(pred_hdr_flat, percentile)
            targ_p = np.percentile(target_hdr_flat, percentile)
            print(f"  P{percentile:2d}: pred={pred_p:6.3f}, target={targ_p:6.3f}")
        
        # Per-channel analysis (compressed space)
        print("\n🌈 PER-CHANNEL ANALYSIS (compressed):")
        for c, color in enumerate(['R', 'G', 'B']):
            pred_c = pred[:, c, :, :]
            targ_c = target[:, c, :, :]
            print(f"  {color}: pred=[{pred_c.min():.3f}, {pred_c.max():.3f}] "
                  f"target=[{targ_c.min():.3f}, {targ_c.max():.3f}]")
        
        # Exposure check
        pred_bright = (pred > 0.5).sum().item()
        pred_dark = (pred < -0.5).sum().item()
        targ_bright = (target > 0.5).sum().item()
        targ_dark = (target < -0.5).sum().item()
        
        print("\n💡 EXPOSURE DISTRIBUTION (compressed space):")
        print(f"  Prediction: {pred_bright} bright (>0.5), {pred_dark} dark (<-0.5) pixels")
        print(f"  Target:     {targ_bright} bright (>0.5), {targ_dark} dark (<-0.5) pixels")
        
        # HDR exposure check (more meaningful)
        pred_hdr_bright = (pred_hdr > 1.0).sum().item()
        target_hdr_bright = (target_hdr > 1.0).sum().item()
        
        print(f"\n🔆 HDR HIGHLIGHTS (>1.0 radiance):")
        print(f"  Prediction: {pred_hdr_bright} pixels")
        print(f"  Target:     {target_hdr_bright} pixels")
        
        # Convert to numpy for PSNR/SSIM calculations
        pred_np = pred[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        target_np = target[0].cpu().numpy().transpose(1, 2, 0)
        pred_hdr_np = pred_hdr[0].cpu().numpy().transpose(1, 2, 0)
        target_hdr_np = target_hdr[0].cpu().numpy().transpose(1, 2, 0)
        
        pred_tonemapped = tonemap(pred_hdr_np)
        target_tonemapped = tonemap(target_hdr_np)
        
        # Compute metrics in different spaces
        print("\n📐 QUALITY METRICS:")
        
        # 1. Compressed space metrics ([-1, 1])
        psnr_compressed = PSNR(range=2.0)
        ssim_compressed = SSIM(range=2.0)
        
        try:
            psnr_comp = psnr_compressed(pred_np, target_np)
            ssim_comp = ssim_compressed(pred_np, target_np)
            print(f"  Compressed space ([-1,1]):")
            print(f"    PSNR: {psnr_comp:.2f} dB")
            print(f"    SSIM: {ssim_comp:.4f}")
        except Exception as e:
            print(f"  Compressed space: Error computing metrics - {e}")
        
        # 2. HDR radiance space metrics (use log for better range)
        # For HDR, we use log-domain PSNR which is more meaningful
        pred_log = np.log10(pred_hdr_np + 1e-6)
        target_log = np.log10(target_hdr_np + 1e-6)
        log_range = max(target_log.max() - target_log.min(), 1.0)
        
        psnr_hdr = PSNR(range=log_range)
        ssim_hdr = SSIM(range=log_range)
        
        try:
            psnr_log = psnr_hdr(pred_log, target_log)
            ssim_log = ssim_hdr(pred_log, target_log)
            print(f"  HDR log-space (meaningful for HDR):")
            print(f"    PSNR: {psnr_log:.2f} dB")
            print(f"    SSIM: {ssim_log:.4f}")
        except Exception as e:
            print(f"  HDR log-space: Error computing metrics - {e}")
        
        # 3. Tonemapped space metrics (traditional perceptual quality)
        psnr_tone = PSNR(range=1.0)
        ssim_tone = SSIM(range=1.0)
        
        try:
            psnr_tm = psnr_tone(pred_tonemapped, target_tonemapped)
            ssim_tm = ssim_tone(pred_tonemapped, target_tonemapped)
            print(f"  Tonemapped LDR (perceptual quality):")
            print(f"    PSNR: {psnr_tm:.2f} dB")
            print(f"    SSIM: {ssim_tm:.4f}")
        except Exception as e:
            print(f"  Tonemapped LDR: Error computing metrics - {e}")
        
        # 4. HDR-VDP-2 style metric (relative MSE in log domain)
        try:
            rel_mse = np.mean((pred_log - target_log) ** 2)
            print(f"  HDR quality:")
            print(f"    Relative MSE (log): {rel_mse:.6f}")
        except Exception as e:
            print(f"  HDR quality: Error computing - {e}")
        
        print(f"{'='*80}\n")


def save_diagnostic_images(pred, target, img1, img2, sample_path, idx=0):
    """Save detailed diagnostic images with proper HDR tonemapping"""
    
    os.makedirs(sample_path, exist_ok=True)
    
    with torch.no_grad():
        # Convert to HDR radiance space
        def to_hdr_radiance(tensor):
            """Convert from compressed [-1,1] to HDR radiance"""
            compressed = (tensor + 1.0) / 2.0  # [-1, 1] -> [0, 1]
            hdr = (torch.pow(1.0 + MU, compressed) - 1.0) / MU
            return hdr
        
        # Convert all tensors to HDR radiance
        pred_hdr = to_hdr_radiance(pred)
        target_hdr = to_hdr_radiance(target)
        img1_hdr = to_hdr_radiance(img1)
        img2_hdr = to_hdr_radiance(img2)
        
        # Convert to numpy
        def hdr_to_displayable(hdr_tensor, tonemap_method='reinhard'):
            """Convert HDR radiance to displayable LDR image"""
            hdr_np = hdr_tensor[0].cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
            
            if tonemap_method == 'reinhard':
                # Reinhard tonemapping
                ldr = hdr_np / (1.0 + hdr_np)
                # Gamma correction
                ldr = np.power(np.clip(ldr, 0, 1), 1.0 / 2.2)
            elif tonemap_method == 'log':
                # Log tonemapping (alternative)
                ldr = np.log(1.0 + hdr_np) / np.log(1.0 + hdr_np.max())
            else:
                # Simple clipping (not recommended but fast)
                ldr = np.clip(hdr_np, 0, 1)
            
            return (ldr * 255).astype(np.uint8)
        
        # Tonemap all images
        pred_img = hdr_to_displayable(pred_hdr)
        target_img = hdr_to_displayable(target_hdr)
        img1_img = hdr_to_displayable(img1_hdr)
        img2_img = hdr_to_displayable(img2_hdr)
        
        # Create error map (in HDR space, then normalize)
        error_hdr = torch.abs(pred_hdr - target_hdr)
        error_max = error_hdr.max()
        if error_max > 0:
            error_normalized = error_hdr / error_max
        else:
            error_normalized = error_hdr
        
        error_img = (error_normalized[0].mean(dim=0).cpu().numpy() * 255).astype(np.uint8)
        error_img = cv2.applyColorMap(error_img, cv2.COLORMAP_JET)
        
        # Create difference image (after tonemapping for fair comparison)
        diff_img = np.abs(pred_img.astype(np.int16) - target_img.astype(np.int16)).astype(np.uint8)
        # Enhance difference for visibility
        diff_img = np.clip(diff_img * 3, 0, 255).astype(np.uint8)
        
        # Create composite image
        h, w = pred_img.shape[:2]
        composite = np.zeros((h * 2, w * 3, 3), dtype=np.uint8)
        
        # Top row: inputs and output
        composite[0:h, 0:w] = img1_img
        composite[0:h, w:2*w] = img2_img
        composite[0:h, 2*w:3*w] = pred_img
        
        # Bottom row: target and errors
        composite[h:2*h, 0:w] = target_img
        composite[h:2*h, w:2*w] = error_img
        composite[h:2*h, 2*w:3*w] = diff_img
        
        # Add labels with background for better visibility
        labels = [
            ("Input 1", (10, 30), (0, h)),
            ("Input 2", (w+10, 30), (0, h)),
            ("Output", (2*w+10, 30), (0, h)),
            ("Target", (10, h+30), (h, 2*h)),
            ("Error (heat)", (w+10, h+30), (h, 2*h)),
            ("Error (diff)", (2*w+10, h+30), (h, 2*h))
        ]
        
        for text, pos, (y1, y2) in labels:
            # Add black background for text
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(composite, 
                         (pos[0]-5, pos[1]-text_size[1]-5),
                         (pos[0]+text_size[0]+5, pos[1]+5),
                         (0, 0, 0), -1)
            cv2.putText(composite, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
        
        # Add statistics overlay
        # Calculate metrics for display
        pred_hdr_np = pred_hdr[0].cpu().numpy().transpose(1, 2, 0)
        target_hdr_np = target_hdr[0].cpu().numpy().transpose(1, 2, 0)
        
        # Tonemap for perceptual metrics
        def simple_tonemap_np(hdr_img):
            tonemapped = hdr_img / (1.0 + hdr_img)
            return np.power(np.clip(tonemapped, 0, 1), 1.0 / 2.2)
        
        pred_tm = simple_tonemap_np(pred_hdr_np)
        target_tm = simple_tonemap_np(target_hdr_np)
        
        # Compute metrics
        psnr_calc = PSNR(range=1.0)
        ssim_calc = SSIM(range=1.0)
        
        try:
            psnr_val = psnr_calc(pred_tm, target_tm)
            ssim_val = ssim_calc(pred_tm, target_tm)
        except:
            psnr_val = 0.0
            ssim_val = 0.0
        
        # HDR log-domain PSNR
        pred_log = np.log10(pred_hdr_np + 1e-6)
        target_log = np.log10(target_hdr_np + 1e-6)
        log_range = max(target_log.max() - target_log.min(), 1.0)
        psnr_hdr = PSNR(range=log_range)
        
        try:
            psnr_hdr_val = psnr_hdr(pred_log, target_log)
        except:
            psnr_hdr_val = 0.0
        
        stats_text = [
            f"PSNR (tone): {psnr_val:.2f} dB",
            f"PSNR (HDR):  {psnr_hdr_val:.2f} dB",
            f"SSIM:        {ssim_val:.4f}",
            f"Pred: [{pred_hdr.min():.3f}, {pred_hdr.max():.3f}]",
            f"Targ: [{target_hdr.min():.3f}, {target_hdr.max():.3f}]",
            f"MAE: {(pred_hdr - target_hdr).abs().mean():.4f}"
        ]
        
        y_offset = h - 140  # Adjusted for more text lines
        for i, stat in enumerate(stats_text):
            text_size = cv2.getTextSize(stat, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.rectangle(composite,
                         (2*w+5, y_offset+i*22-text_size[1]-3),
                         (2*w+text_size[0]+10, y_offset+i*22+3),
                         (0, 0, 0), -1)
            cv2.putText(composite, stat, (2*w+10, y_offset+i*22),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        
        # Save
        save_path = os.path.join(sample_path, f'diagnostic_{idx:04d}.png')
        cv2.imwrite(save_path, composite[..., ::-1])  # RGB -> BGR
        print(f"    💾 Saved diagnostic: {save_path}")
        
        


def diagnose_model_and_data(model, img1, img2, sum1, sum2, target):
    """Comprehensive diagnostic before training"""
    
    print("="*80)
    print("🔍 PRE-TRAINING DIAGNOSTICS")
    print("="*80)
    
    # 1. Check input data ranges
    print("\n📊 INPUT DATA RANGES (should be close to [-1, 1]):")
    print(f"  img1:   [{img1.min():.3f}, {img1.max():.3f}]", 
          "⚠️ EXCEEDS [-1,1]" if img1.min() < -1.1 or img1.max() > 1.1 else "✓")
    print(f"  img2:   [{img2.min():.3f}, {img2.max():.3f}]",
          "⚠️ EXCEEDS [-1,1]" if img2.min() < -1.1 or img2.max() > 1.1 else "✓")
    print(f"  sum1:   [{sum1.min():.3f}, {sum1.max():.3f}]",
          "⚠️ EXCEEDS [-1,1]" if sum1.min() < -1.1 or sum1.max() > 1.1 else "✓")
    print(f"  sum2:   [{sum2.min():.3f}, {sum2.max():.3f}]",
          "⚠️ EXCEEDS [-1,1]" if sum2.min() < -1.1 or sum2.max() > 1.1 else "✓")
    print(f"  target: [{target.min():.3f}, {target.max():.3f}]",
          "⚠️ EXCEEDS [-1,1]" if target.min() < -1.1 or target.max() > 1.1 else "✓")
    
    # 2. Check model forward pass
    print("\n🔧 MODEL FORWARD PASS:")
    model.eval()
    with torch.no_grad():
        try:
            output = model(img1, img2, sum1, sum2)
            print(f"  Output shape: {output.shape} ✓")
            print(f"  Output range: [{output.min():.3f}, {output.max():.3f}]", end=" ")
            
            if output.min() < -1.0 or output.max() > 1.0:
                print("⚠️ EXCEEDS [-1,1] (tanh should prevent this!)")
            elif output.min() > -0.5 and output.max() < 0.5:
                print("⚠️ TOO NARROW (model is timid)")
            else:
                print("✓")
            
            if torch.isnan(output).any():
                print("  ❌ NaN detected in output!")
            if torch.isinf(output).any():
                print("  ❌ Inf detected in output!")
                
        except Exception as e:
            print(f"  ❌ Forward pass failed: {e}")
            return
    
    # 3. Check dynamic range
    print("\n🌟 DYNAMIC RANGE ANALYSIS:")
    output_hdr = inverse_transform(output)
    target_hdr = inverse_transform(target)
    
    output_dr = output_hdr.max() / (output_hdr.min() + 1e-8)
    target_dr = target_hdr.max() / (target_hdr.min() + 1e-8)
    dr_ratio = output_dr / target_dr
    
    print(f"  Output DR:  {output_dr:.1f}:1")
    print(f"  Target DR:  {target_dr:.1f}:1")
    print(f"  Ratio:      {dr_ratio:.4f}", end=" ")
    
    if dr_ratio < 0.5:
        print("❌ SEVERELY COMPRESSED")
    elif dr_ratio < 0.8:
        print("⚠️ UNDER-COMPRESSED")
    elif dr_ratio > 1.2:
        print("⚠️ OVER-EXPANDED")
    else:
        print("✓")
    
    # 4. Check gradient flow (single backward pass)
    print("\n🔄 GRADIENT FLOW CHECK:")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    output = model(img1, img2, sum1, sum2)
    loss = torch.nn.functional.l1_loss(output, target)
    
    optimizer.zero_grad()
    loss.backward()
    
    total_norm = 0.0
    num_params = 0
    zero_grad_params = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2).item()
            total_norm += param_norm ** 2
            num_params += 1
            
            if param_norm < 1e-10:
                zero_grad_params += 1
    
    total_norm = total_norm ** 0.5
    
    print(f"  Total gradient norm: {total_norm:.6f}", end=" ")
    if total_norm < 1e-6:
        print("❌ VANISHING GRADIENTS")
    elif total_norm > 100:
        print("⚠️ EXPLODING GRADIENTS")
    else:
        print("✓")
    
    print(f"  Zero-gradient params: {zero_grad_params}/{num_params}", end=" ")
    if zero_grad_params > num_params * 0.1:
        print("⚠️ TOO MANY")
    else:
        print("✓")
    
    # 5. Check learnable parameters
    print("\n⚙️  LEARNABLE PARAMETERS:")
    if hasattr(model, 'output_scale'):
        print(f"  output_scale: {model.output_scale.item():.4f}")
        print(f"  output_bias:  {model.output_bias.item():.4f}")
    else:
        print("  ⚠️ No output_scale/bias parameters (using old model)")
    
    # 6. Initial loss value
    print("\n💰 INITIAL LOSS:")
    initial_loss = loss.item()
    print(f"  L1 Loss: {initial_loss:.6f}", end=" ")
    
    if initial_loss > 1.0:
        print("⚠️ VERY HIGH (check normalization)")
    elif initial_loss < 0.001:
        print("⚠️ SUSPICIOUSLY LOW")
    else:
        print("✓")
    
    # 7. Memory check
    print("\n💾 GPU MEMORY:")
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  Allocated: {allocated:.2f} GB")
        print(f"  Reserved:  {reserved:.2f} GB")
    else:
        print("  Running on CPU")
    
    # 8. Quick overfitting test
    print("\n🚀 QUICK OVERFITTING TEST (10 steps):")
    losses = []
    for step in range(10):
        output = model(img1, img2, sum1, sum2)
        loss = torch.nn.functional.l1_loss(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    loss_decrease = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"  Initial loss:  {losses[0]:.6f}")
    print(f"  Final loss:    {losses[-1]:.6f}")
    print(f"  Decrease:      {loss_decrease:.2f}%", end=" ")
    
    if loss_decrease < 1:
        print("❌ NOT LEARNING (check LR, gradients, architecture)")
    elif loss_decrease < 10:
        print("⚠️ SLOW LEARNING (may need higher LR)")
    else:
        print("✓")
    
    print("\n" + "="*80)
    print("💡 RECOMMENDATIONS:")
    print("="*80)
    
    recommendations = []
    
    # Input range issues
    if img1.min() < -1.1 or img1.max() > 1.1:
        recommendations.append("❌ Fix img1 normalization in transform()")
    
    # Dynamic range issues
    if dr_ratio < 0.8:
        recommendations.append("⚠️ Increase range_weight in loss (try 5.0)")
        recommendations.append("⚠️ Check if output_scale parameter exists")
    
    # Gradient issues
    if total_norm < 1e-6:
        recommendations.append("❌ Vanishing gradients - increase learning rate")
    elif total_norm > 100:
        recommendations.append("⚠️ Add gradient clipping (max_norm=1.0)")
    
    # Learning issues
    if loss_decrease < 10:
        recommendations.append("⚠️ Increase learning rate (try 5e-4 or 1e-3)")
        recommendations.append("⚠️ Reduce regularization (weight_decay)")
    
    # Output issues
    if output.min() > -0.5 and output.max() < 0.5:
        recommendations.append("⚠️ Model is timid - use new model with output scaling")
    
    if len(recommendations) == 0:
        print("✅ Everything looks good! Ready to train.")
    else:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    
    print("="*80 + "\n")
    
    return {
        'initial_loss': initial_loss,
        'dr_ratio': dr_ratio,
        'grad_norm': total_norm,
        'loss_decrease': loss_decrease,
        'output_range': (output.min().item(), output.max().item())
    }
