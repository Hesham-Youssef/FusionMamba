import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utils.tools import LDR2HDR, GAMMA, MU


def diagnose_exposure_conversion():
    """Deep dive into LDR->HDR conversion with different exposures"""
    print("\n" + "="*80)
    print("EXPOSURE CONVERSION DEEP DIAGNOSIS")
    print("="*80)
    print(f"Using GAMMA={GAMMA}, MU={MU}")
    
    # Create test LDR images with known values
    test_values = [0.0, 0.25, 0.5, 0.75, 1.0]  # LDR values [0, 1]
    print(f"\nTesting with LDR pixel values: {test_values}")
    
    # Test different exposure ratios
    expo_pairs = [
        (0.5, 2.0),   # 4x difference
        (1.0, 4.0),   # 4x difference
        (0.25, 1.0),  # 4x difference
        (1.0, 2.0),   # 2x difference
        (1.0, 1.0),   # No difference (should fail!)
    ]
    
    for expo1, expo2 in expo_pairs:
        print(f"\n--- Exposure Pair: {expo1:.2f} vs {expo2:.2f} (ratio: {expo2/expo1:.2f}x) ---")
        
        for val in test_values:
            # Convert to [-1, 1] range as expected by LDR2HDR
            ldr_norm = val * 2 - 1
            
            # Convert to HDR with both exposures
            hdr1 = LDR2HDR(np.array([[[ldr_norm, ldr_norm, ldr_norm]]]), expo1)
            hdr2 = LDR2HDR(np.array([[[ldr_norm, ldr_norm, ldr_norm]]]), expo2)
            
            hdr1_val = hdr1[0, 0, 0]
            hdr2_val = hdr2[0, 0, 0]
            
            ratio = hdr1_val / (hdr2_val + 1e-8)
            expected_ratio = expo1 / expo2
            
            print(f"  LDR={val:.2f} -> HDR1={hdr1_val:.4f}, HDR2={hdr2_val:.4f}, "
                  f"ratio={ratio:.4f} (expected={expected_ratio:.4f})")
    
    print("\n" + "="*80)


def diagnose_data_batch(data_tuple, save_path='data_diagnosis.png'):
    """Comprehensive diagnosis of a data batch"""
    
    # FIXED: Dataset returns 5 values, not 7!
    if len(data_tuple) == 5:
        img1_hdr, img2_hdr, sum1, sum2, ref_HDR = data_tuple
        print("\n✓ Correct unpacking: 5 values from dataset")
    else:
        print(f"\n❌ ERROR: Expected 5 values, got {len(data_tuple)}")
        return [], []
    
    print("\n" + "="*80)
    print("DATA BATCH DIAGNOSIS")
    print("="*80)
    
    # Convert to numpy
    img1_np = img1_hdr[0].cpu().numpy().transpose(1, 2, 0)
    img2_np = img2_hdr[0].cpu().numpy().transpose(1, 2, 0)
    ref_np = ref_HDR[0].cpu().numpy().transpose(1, 2, 0)
    sum1_np = sum1[0].cpu().numpy().transpose(1, 2, 0)
    sum2_np = sum2[0].cpu().numpy().transpose(1, 2, 0)
    
    # DETAILED STATISTICS
    print("\nDETAILED STATISTICS:")
    for name, data in [("img1_hdr", img1_hdr), ("img2_hdr", img2_hdr), 
                        ("sum1", sum1), ("sum2", sum2), ("ref_HDR", ref_HDR)]:
        print(f"\n{name}:")
        print(f"  Shape: {data.shape}")
        print(f"  Range: [{data.min():.6f}, {data.max():.6f}]")
        print(f"  Mean: {data.mean():.6f}, Std: {data.std():.6f}")
        print(f"  Per-channel means: R={data[0,0].mean():.6f}, "
              f"G={data[0,1].mean():.6f}, B={data[0,2].mean():.6f}")
        
        # Check for suspicious patterns
        if torch.allclose(data[0,0], data[0,1], atol=1e-5):
            print(f"  ⚠️  WARNING: R and G channels identical!")
        if torch.allclose(data[0,0], data[0,2], atol=1e-5):
            print(f"  ⚠️  WARNING: R and B channels identical!")
    
    # CHECK SUMMARY FORMAT
    print("\n" + "="*80)
    print("SUMMARY FORMAT CHECK:")
    print("="*80)
    
    # Summaries should be in [-1, 1] range (LDR format)
    # But the code should tonemap img1_hdr and compare with sum1
    print(f"sum1 range: [{sum1.min():.6f}, {sum1.max():.6f}]")
    print(f"sum2 range: [{sum2.min():.6f}, {sum2.max():.6f}]")
    
    if sum1.min() >= -1.1 and sum1.max() <= 1.1:
        print("✓ Summaries appear to be in LDR format [-1, 1]")
    else:
        print("⚠️  Summaries might be in unexpected format")
    
    # EXPOSURE DIFFERENCE ANALYSIS
    print("\n" + "="*80)
    print("EXPOSURE DIFFERENCE ANALYSIS:")
    print("="*80)
    
    diff_abs = torch.abs(img1_hdr - img2_hdr).mean().item()
    diff_rel = (img1_hdr - img2_hdr).abs().mean() / (img1_hdr.abs().mean() + 1e-8)
    
    print(f"Absolute difference: {diff_abs:.6f}")
    print(f"Relative difference: {diff_rel:.6f}")
    
    # Compute per-pixel ratio
    ratio_map = img1_hdr / (img2_hdr + 1e-8)
    ratio_mean = ratio_map.mean().item()
    ratio_std = ratio_map.std().item()
    
    print(f"Mean pixel ratio (img1/img2): {ratio_mean:.4f}")
    print(f"Std of pixel ratios: {ratio_std:.4f}")
    
    if diff_abs < 0.01:
        print("❌ CRITICAL: Inputs are nearly IDENTICAL!")
        print("   This means exposure difference is NOT being applied!")
    elif diff_abs < 0.05:
        print("⚠️  WARNING: Inputs are very similar (small exposure difference)")
    else:
        print("✓ Inputs are sufficiently different")
    
    # INTENSITY ANALYSIS
    print("\n" + "="*80)
    print("INTENSITY ANALYSIS:")
    print("="*80)
    
    mean_ratio = img1_hdr.mean() / (img2_hdr.mean() + 1e-8)
    print(f"Mean intensity ratio: {mean_ratio:.4f}")
    
    if 0.95 < abs(mean_ratio) < 1.05:
        print("❌ CRITICAL: Mean intensities too similar!")
    
    # VISUALIZATION
    img1_display = np.clip((img1_np + 1) / 2, 0, 1)
    img2_display = np.clip((img2_np + 1) / 2, 0, 1)
    ref_display = np.clip((ref_np + 1) / 2, 0, 1)

    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    # Row 1: HDR inputs (already log-compressed, just scaled for display)
    axes[0, 0].imshow(img1_display)
    axes[0, 0].set_title(f'img1_hdr (μ={img1_hdr.mean():.3f})')
    axes[0, 0].axis('off')
    
    axes[0, 0].imshow(img2_display)
    axes[0, 0].set_title(f'img1_hdr (μ={img1_hdr.mean():.3f})')
    axes[0, 1].axis('off')
    
    axes[0, 0].imshow(ref_display)
    axes[0, 0].set_title(f'img1_hdr (μ={img1_hdr.mean():.3f})')
    axes[0, 2].axis('off')
    
    input_diff = np.abs(img1_np - img2_np)
    im = axes[0, 3].imshow(np.clip((input_diff + 1) / 2, 0, 1), cmap='hot', vmin=0, vmax=1)
    axes[0, 3].set_title(f'|img1-img2| (max={input_diff.max():.3f})')
    axes[0, 3].axis('off')
    plt.colorbar(im, ax=axes[0, 3], fraction=0.046)
    
    # Row 2: Sum images
    axes[1, 0].imshow(np.clip((sum1_np + 1) / 2, 0, 1))
    axes[1, 0].set_title(f'sum1 (μ={sum1.mean():.3f})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(np.clip((sum2_np + 1) / 2, 0, 1))
    axes[1, 1].set_title(f'sum2 (μ={sum2.mean():.3f})')
    axes[1, 1].axis('off')
    
    # Ratio map
    ratio_vis = np.clip(ratio_map[0].cpu().numpy().transpose(1, 2, 0), -5, 5)
    im = axes[1, 2].imshow(ratio_vis[:,:,0], cmap='RdBu_r', vmin=-2, vmax=2)
    axes[1, 2].set_title(f'Ratio img1/img2 (μ={ratio_mean:.2f})')
    axes[1, 2].axis('off')
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046)
    
    # Histogram comparison
    axes[1, 3].hist(img1_np.ravel(), bins=50, alpha=0.5, label='img1', color='red', density=True)
    axes[1, 3].hist(img2_np.ravel(), bins=50, alpha=0.5, label='img2', color='blue', density=True)
    axes[1, 3].hist(ref_np.ravel(), bins=50, alpha=0.5, label='ref', color='green', density=True)
    axes[1, 3].set_xlabel('HDR Value')
    axes[1, 3].set_ylabel('Density')
    axes[1, 3].set_title('HDR Value Distributions')
    axes[1, 3].legend()
    axes[1, 3].set_xlim(-2, 2)
    axes[1, 3].grid(True, alpha=0.3)
    
    # Row 3: Per-channel analysis
    for i, (color, name) in enumerate([(0, 'Red'), (1, 'Green'), (2, 'Blue')]):
        if i < 3:
            axes[2, i].hist(img1_np[:,:,color].ravel(), bins=50, alpha=0.5, 
                           label='img1', color='red', density=True)
            axes[2, i].hist(img2_np[:,:,color].ravel(), bins=50, alpha=0.5, 
                           label='img2', color='blue', density=True)
            axes[2, i].hist(ref_np[:,:,color].ravel(), bins=50, alpha=0.5, 
                           label='ref', color='green', density=True)
            axes[2, i].set_title(f'{name} Channel')
            axes[2, i].set_xlabel('Value')
            axes[2, i].set_ylabel('Density')
            axes[2, i].legend()
            axes[2, i].set_xlim(-2, 2)
            axes[2, i].grid(True, alpha=0.3)
    
    # Summary statistics box
    summary_text = f"""SUMMARY:
    
Input Difference: {diff_abs:.6f}
Mean Ratio: {mean_ratio:.4f}
    
img1 range: [{img1_hdr.min():.3f}, {img1_hdr.max():.3f}]
img2 range: [{img2_hdr.min():.3f}, {img2_hdr.max():.3f}]
ref range: [{ref_HDR.min():.3f}, {ref_HDR.max():.3f}]

sum1 range: [{sum1.min():.3f}, {sum1.max():.3f}]
sum2 range: [{sum2.min():.3f}, {sum2.max():.3f}]
    """
    axes[2, 3].text(0.1, 0.5, summary_text, fontsize=9, 
                    family='monospace', verticalalignment='center')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved to {save_path}")
    plt.close()
    
    # CRITICAL CHECKS
    print("\n" + "="*80)
    print("CRITICAL ISSUE DETECTION:")
    print("="*80)
    
    issues = []
    warnings = []
    
    # Check 1: Input difference
    if diff_abs < 0.01:
        issues.append("❌ CRITICAL: Inputs nearly identical - exposure not applied!")
    elif diff_abs < 0.05:
        warnings.append(f"⚠️  Inputs very similar (diff={diff_abs:.4f})")
    else:
        print(f"✓ Input difference OK (diff={diff_abs:.4f})")
    
    # Check 2: Dynamic range
    ref_range = (ref_HDR.max() - ref_HDR.min()).item()
    if ref_range < 0.3:
        issues.append(f"❌ CRITICAL: Target has very low dynamic range ({ref_range:.4f})")
    elif ref_range < 0.5:
        warnings.append(f"⚠️  Target has low dynamic range ({ref_range:.4f})")
    else:
        print(f"✓ Target dynamic range OK ({ref_range:.4f})")
    
    # Check 3: Mean values
    if abs(ref_HDR.mean()) > 0.8:
        warnings.append(f"⚠️  Target mean extreme ({ref_HDR.mean():.4f})")
    else:
        print(f"✓ Target mean reasonable ({ref_HDR.mean():.4f})")
    
    # Check 4: Channel variance
    for i, c in enumerate(['R', 'G', 'B']):
        ch_std = ref_HDR[0, i].std().item()
        if ch_std < 0.05:
            issues.append(f"❌ {c} channel has very low variance ({ch_std:.4f})")
        elif ch_std < 0.1:
            warnings.append(f"⚠️  {c} channel has low variance ({ch_std:.4f})")
    
    # Check 5: Exposure ratio consistency
    if 0.9 < abs(mean_ratio) < 1.1:
        issues.append(f"❌ CRITICAL: Mean intensity ratio too close to 1.0 ({mean_ratio:.4f})")
    else:
        print(f"✓ Mean intensity ratio OK ({mean_ratio:.4f})")
    
    # Check 6: NaN or Inf
    if torch.isnan(ref_HDR).any() or torch.isinf(ref_HDR).any():
        issues.append("❌ CRITICAL: NaN or Inf values detected!")
    
    if issues:
        print("\n" + "!"*80)
        print("CRITICAL ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
        print("!"*80)
    
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  {warning}")
    
    if not issues and not warnings:
        print("\n✅ All checks passed!")
    
    print("="*80 + "\n")
    
    return issues, warnings


def check_exposure_values(configs):
    """Check actual exposure values in the dataset"""
    print("\n" + "="*80)
    print("CHECKING ACTUAL EXPOSURE VALUES IN DATASET")
    print("="*80)
    
    import os
    from glob import glob
    
    filepath = os.path.join(configs.data_path, 'train')
    scene_dirs = [sd for sd in os.listdir(filepath) if os.path.isdir(os.path.join(filepath, sd))]
    
    print(f"\nFound {len(scene_dirs)} scenes")
    print("\nChecking first 3 scenes:")
    
    for i, scene_dir in enumerate(scene_dirs[:3]):
        print(f"\n--- Scene {i+1}: {scene_dir} ---")
        scene_path = os.path.join(filepath, scene_dir)
        exp_path = os.path.join(scene_path, 'input_exp.txt')
        
        if os.path.exists(exp_path):
            exp_raw = open(exp_path).read().split('\n')[:configs.num_shots]
            exp_values = np.array(exp_raw).astype(np.float32)
            exp_multipliers = 2 ** exp_values
            
            print(f"  Raw exposure values: {exp_values}")
            print(f"  Exposure multipliers (2^exp): {exp_multipliers}")
            print(f"  Ratio (max/min): {exp_multipliers.max() / exp_multipliers.min():.4f}")
            
            if len(np.unique(exp_values)) == 1:
                print(f"  ❌ CRITICAL: All exposures are IDENTICAL!")
            elif exp_multipliers.max() / exp_multipliers.min() < 1.5:
                print(f"  ⚠️  WARNING: Exposure variation is small")
            else:
                print(f"  ✓ Exposure variation looks good")
        else:
            print(f"  ❌ ERROR: Exposure file not found!")
    
    print("="*80)


def test_data_loading_comprehensive():
    """Comprehensive data loading test"""
    from config import Configs
    from utils.hdr_load_train_data import U2NetDataset
    from torch.utils.data import DataLoader
    
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA LOADING TEST")
    print("="*80)
    
    configs = Configs()
    print(f"\nDataset configuration:")
    print(f"  Train path: {configs.data_path}")
    print(f"  Patch size: {configs.patch_size}")
    print(f"  Num shots: {configs.num_shots}")
    
    # Check exposure values first
    check_exposure_values(configs)
    
    dataset = U2NetDataset(configs=configs)
    print(f"\n  Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Test multiple batches
    print("\n" + "-"*80)
    print("Testing 10 random batches...")
    print("-"*80)
    
    all_issues = []
    all_warnings = []
    
    for batch_idx in range(10):
        print(f"\n>>> BATCH {batch_idx + 1}/10 <<<")
        data = next(iter(dataloader))
        
        # Correct unpacking for 5 values
        img1_hdr, img2_hdr, sum1, sum2, ref_HDR = data
        
        # Quick stats
        diff = torch.abs(img1_hdr - img2_hdr).mean().item()
        ratio = (img1_hdr.mean() / (img2_hdr.mean() + 1e-8)).item()
        ref_range = (ref_HDR.max() - ref_HDR.min()).item()
        
        status = "✓ OK"
        if diff < 0.01:
            status = "❌ FAIL (identical inputs)"
            all_issues.append(f"Batch {batch_idx+1}: identical inputs")
        elif 0.9 < abs(ratio) < 1.1:
            status = "⚠️  WARN (similar intensities)"
            all_warnings.append(f"Batch {batch_idx+1}: similar intensities")
        
        print(f"  Input diff: {diff:.6f}, Ratio: {ratio:.4f}, "
              f"Ref range: {ref_range:.4f} - {status}")
    
    # Detailed diagnosis on first batch
    print("\n" + "="*80)
    print("DETAILED DIAGNOSIS OF FIRST BATCH")
    print("="*80)
    
    data = next(iter(dataloader))
    issues, warnings = diagnose_data_batch(data, save_path='data_diagnosis_detailed.png')
    
    all_issues.extend(issues)
    all_warnings.extend(warnings)
    
    # FINAL SUMMARY
    print("\n" + "="*80)
    print("FINAL DIAGNOSTIC SUMMARY")
    print("="*80)
    
    if all_issues:
        print(f"\n❌ Found {len(all_issues)} CRITICAL issues:")
        for issue in set(all_issues):
            print(f"  • {issue}")
        
        print("\n🔧 RECOMMENDED FIXES:")
        if any("identical" in str(i).lower() for i in all_issues):
            print("  1. Check exposure file (input_exp.txt) - ensure different exposures")
            print("  2. Verify LDR2HDR is being called with DIFFERENT exposures")
            print("  3. Check that idx1 != idx2 in dataset __getitem__")
            print("  4. Print actual exposure values being used")
        if any("dynamic range" in str(i).lower() for i in all_issues):
            print("  5. Verify HDR reference images are loaded correctly")
            print("  6. Check normalization and scaling")
    
    if all_warnings:
        print(f"\n⚠️  Found {len(all_warnings)} warnings:")
        for warning in set(all_warnings):
            print(f"  • {warning}")
    
    if not all_issues and not all_warnings:
        print("\n✅ ALL TESTS PASSED - Data loading appears correct!")
    
    print("="*80 + "\n")
    
    return all_issues, all_warnings


if __name__ == '__main__':
    print("="*80)
    print("ENHANCED HDR DATA LOADING DIAGNOSTIC TOOL")
    print("="*80)
    print(f"Using GAMMA={GAMMA}, MU={MU}")
    
    # Test 1: Exposure conversion
    diagnose_exposure_conversion()
    
    # Test 2: Comprehensive data loading
    issues, warnings = test_data_loading_comprehensive()
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    
    if issues:
        print("\n⚠️  Issues found - review output above for details")
        print("Check 'data_diagnosis_detailed.png' for visualizations")
    else:
        print("\n✅ No critical issues detected!")