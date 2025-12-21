"""
Comprehensive HDR Output Diagnostic Tool
Run this to diagnose poor output quality with detailed analysis
"""

import torch
import numpy as np
import cv2
import os
from utils.tools import tonemap, inverse_transform, tonemap_np, HDRLoss
from model.u2net import U2Net
from config import Configs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from glob import glob
import pickle
import json


def plot_histogram(data, title, filename, log_scale=False):
    """Plot histogram of values"""
    plt.figure(figsize=(10, 6))
    flat_data = data.flatten()
    
    # Remove inf and nan
    flat_data = flat_data[np.isfinite(flat_data)]
    
    if len(flat_data) > 0:
        plt.hist(flat_data, bins=100, alpha=0.7, edgecolor='black')
        plt.title(title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        if log_scale:
            plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close()
        return True
    return False


def analyze_hdr_output(hdr_path):
    """Comprehensive analysis of a saved HDR file"""
    print("\n" + "="*80)
    print(f"ANALYZING HDR OUTPUT: {os.path.basename(hdr_path)}")
    print("="*80)
    
    if not os.path.exists(hdr_path):
        print(f"❌ File not found: {hdr_path}")
        return None
    
    # Read the HDR file
    img = cv2.imread(hdr_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"❌ Could not read HDR file")
        return None
    
    img = img[..., ::-1]  # BGR to RGB
    
    # Basic statistics
    print(f"\n📊 BASIC STATISTICS:")
    print(f"  Shape: {img.shape}")
    print(f"  Data type: {img.dtype}")
    print(f"  Size: {img.size:,} values ({img.nbytes / 1024:.2f} KB)")
    
    print(f"\n📈 VALUE RANGE:")
    print(f"  Min value:     {np.min(img):>12.6f}")
    print(f"  Max value:     {np.max(img):>12.6f}")
    print(f"  Mean value:    {np.mean(img):>12.6f}")
    print(f"  Median value:  {np.median(img):>12.6f}")
    print(f"  Std dev:       {np.std(img):>12.6f}")
    print(f"  1st percentile:  {np.percentile(img, 1):>12.6f}")
    print(f"  99th percentile: {np.percentile(img, 99):>12.6f}")
    
    print(f"\n🔍 DATA QUALITY:")
    nan_count = np.isnan(img).sum()
    inf_count = np.isinf(img).sum()
    zero_count = (img == 0).sum()
    neg_count = (img < 0).sum()
    
    print(f"  NaN values:      {nan_count:>10} ({100*nan_count/img.size:.2f}%)")
    print(f"  Inf values:      {inf_count:>10} ({100*inf_count/img.size:.2f}%)")
    print(f"  Zero values:     {zero_count:>10} ({100*zero_count/img.size:.2f}%)")
    print(f"  Negative values: {neg_count:>10} ({100*neg_count/img.size:.2f}%)")
    
    # Detailed value distribution
    print(f"\n📊 VALUE DISTRIBUTION:")
    ranges = [
        ("< 0", img < 0),
        ("= 0", img == 0),
        ("> 0 and < 0.001", (img > 0) & (img < 0.001)),
        (">= 0.001 and < 0.01", (img >= 0.001) & (img < 0.01)),
        (">= 0.01 and < 0.1", (img >= 0.01) & (img < 0.1)),
        (">= 0.1 and < 1", (img >= 0.1) & (img < 1)),
        (">= 1 and < 10", (img >= 1) & (img < 10)),
        (">= 10", img >= 10),
    ]
    
    for label, mask in ranges:
        count = mask.sum()
        print(f"  {label:>25}: {count:>10} ({100*count/img.size:>6.2f}%)")
    
    # Per-channel analysis
    print(f"\n🎨 PER-CHANNEL ANALYSIS:")
    for i, channel in enumerate(['R', 'G', 'B']):
        ch_data = img[:, :, i]
        print(f"  {channel} channel:")
        print(f"    Min: {ch_data.min():>10.6f}  Max: {ch_data.max():>10.6f}  Mean: {ch_data.mean():>10.6f}")
    
    # Check for clipping/saturation
    print(f"\n⚠️  POTENTIAL ISSUES:")
    issues = []
    
    if np.mean(img) < 0.001:
        issues.append("Image is extremely dark (mean < 0.001)")
    if np.percentile(img, 99) < 0.01:
        issues.append("99th percentile < 0.01 - image severely underexposed")
    if zero_count / img.size > 0.5:
        issues.append(f"{100*zero_count/img.size:.1f}% of pixels are exactly zero")
    if np.max(img) < 0.1:
        issues.append(f"Maximum value is only {np.max(img):.6f} - very dark output")
    if nan_count > 0:
        issues.append(f"{nan_count} NaN values detected")
    if inf_count > 0:
        issues.append(f"{inf_count} infinite values detected")
    if neg_count > img.size * 0.01:
        issues.append(f"{100*neg_count/img.size:.1f}% negative values (HDR should be positive)")
    
    # Check for collapsed output
    if np.std(img) < 0.001:
        issues.append("Output has collapsed (std dev < 0.001)")
    
    # Check if output is constant
    unique_vals = len(np.unique(img.flatten()))
    if unique_vals < 10:
        issues.append(f"Only {unique_vals} unique values - output may be collapsed")
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. ❌ {issue}")
    else:
        print(f"  ✅ No obvious issues detected")
    
    # Create visualizations
    print(f"\n🎨 CREATING VISUALIZATIONS...")
    
    os.makedirs('debug_output', exist_ok=True)
    
    # Histogram
    if plot_histogram(img, 'HDR Value Distribution', 'debug_output/histogram.png'):
        print(f"  ✅ Saved: debug_output/histogram.png")
    
    if plot_histogram(img, 'HDR Value Distribution (log scale)', 'debug_output/histogram_log.png', log_scale=True):
        print(f"  ✅ Saved: debug_output/histogram_log.png")
    
    # Multiple tone mapping methods
    tone_methods = []
    
    # Method 1: Simple gamma
    try:
        tonemap_simple = np.clip(img ** (1/2.2), 0, 1)
        cv2.imwrite('debug_output/tonemap_simple.png', (tonemap_simple * 255).astype(np.uint8)[..., ::-1])
        tone_methods.append(('simple', tonemap_simple))
        print(f"  ✅ Saved: debug_output/tonemap_simple.png")
    except Exception as e:
        print(f"  ❌ Simple tonemap failed: {e}")
    
    # Method 2: Log
    try:
        tonemap_log = np.log1p(img * 10) / np.log1p(10)
        tonemap_log = np.clip(tonemap_log, 0, 1)
        cv2.imwrite('debug_output/tonemap_log.png', (tonemap_log * 255).astype(np.uint8)[..., ::-1])
        tone_methods.append(('log', tonemap_log))
        print(f"  ✅ Saved: debug_output/tonemap_log.png")
    except Exception as e:
        print(f"  ❌ Log tonemap failed: {e}")
    
    # Method 3: Mu-law (your method)
    try:
        tonemap_mu = tonemap_np(img * 2 - 1)
        tonemap_mu = (tonemap_mu + 1) / 2
        tonemap_mu = np.clip(tonemap_mu, 0, 1)
        cv2.imwrite('debug_output/tonemap_mu.png', (tonemap_mu * 255).astype(np.uint8)[..., ::-1])
        tone_methods.append(('mu-law', tonemap_mu))
        print(f"  ✅ Saved: debug_output/tonemap_mu.png")
    except Exception as e:
        print(f"  ❌ Mu-law tonemap failed: {e}")
    
    # Method 4: Auto-exposure
    try:
        percentile_99 = np.percentile(img, 99)
        if percentile_99 > 1e-6:
            tonemap_auto = np.clip(img / percentile_99, 0, 1) ** (1/2.2)
            cv2.imwrite('debug_output/tonemap_auto.png', (tonemap_auto * 255).astype(np.uint8)[..., ::-1])
            tone_methods.append(('auto-exposure', tonemap_auto))
            print(f"  ✅ Saved: debug_output/tonemap_auto.png")
    except Exception as e:
        print(f"  ❌ Auto-exposure tonemap failed: {e}")
    
    # Method 5: Reinhard
    try:
        L = 0.2126 * img[:,:,0] + 0.7152 * img[:,:,1] + 0.0722 * img[:,:,2]
        L_white = np.percentile(L, 99)
        if L_white > 1e-6:
            tonemap_reinhard = img * (1 + L[:,:,None] / (L_white**2)) / (1 + L[:,:,None])
            tonemap_reinhard = np.clip(tonemap_reinhard ** (1/2.2), 0, 1)
            cv2.imwrite('debug_output/tonemap_reinhard.png', (tonemap_reinhard * 255).astype(np.uint8)[..., ::-1])
            tone_methods.append(('reinhard', tonemap_reinhard))
            print(f"  ✅ Saved: debug_output/tonemap_reinhard.png")
    except Exception as e:
        print(f"  ❌ Reinhard tonemap failed: {e}")
    
    # Method 6: Linear stretch
    try:
        if img.max() > img.min():
            tonemap_stretch = (img - img.min()) / (img.max() - img.min())
            cv2.imwrite('debug_output/tonemap_stretch.png', (tonemap_stretch * 255).astype(np.uint8)[..., ::-1])
            tone_methods.append(('stretch', tonemap_stretch))
            print(f"  ✅ Saved: debug_output/tonemap_stretch.png")
    except Exception as e:
        print(f"  ❌ Stretch tonemap failed: {e}")
    
    return img, tone_methods


def check_model_architecture():
    """Analyze model architecture for potential issues"""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE ANALYSIS")
    print("="*80)
    
    try:
        configs = Configs()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        dim = configs.dim if hasattr(configs, 'dim') else 32
        img1_dim = configs.c_dim * 2
        img2_dim = configs.c_dim * 2
        H = configs.patch_size[0] if hasattr(configs, 'patch_size') else 64
        W = configs.patch_size[1] if hasattr(configs, 'patch_size') else 64
        
        model = U2Net(dim=dim, img1_dim=img1_dim, img2_dim=img2_dim, H=H, W=W)
        model.to(device)
        
        print(f"\n📐 MODEL CONFIGURATION:")
        print(f"  Dimension: {dim}")
        print(f"  Input 1 channels: {img1_dim}")
        print(f"  Input 2 channels: {img2_dim}")
        print(f"  Patch size: {H}x{W}")
        
        # Check output layer
        print(f"\n🔍 OUTPUT LAYER ANALYSIS:")
        # Try to find the final layer
        last_layer_name = None
        last_layer = None
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                last_layer_name = name
                last_layer = module
        
        if last_layer is not None:
            print(f"  Last layer: {last_layer_name}")
            print(f"  Type: {type(last_layer).__name__}")
            
            # Check for activation functions
            if isinstance(last_layer, (torch.nn.ReLU, torch.nn.LeakyReLU, torch.nn.ELU)):
                print(f"  ⚠️  WARNING: Output has {type(last_layer).__name__} activation!")
                print(f"     This will clip negative values and may cause issues for HDR")
            elif isinstance(last_layer, torch.nn.Sigmoid):
                print(f"  ⚠️  WARNING: Output has Sigmoid activation!")
                print(f"     This limits output to [0, 1] which is wrong for HDR [-1, 1]")
            elif isinstance(last_layer, torch.nn.Tanh):
                print(f"  ✅ Output has Tanh activation (good for [-1, 1] range)")
            else:
                print(f"  ℹ️  No explicit activation on output layer")
        
        # Count parameters by layer type
        print(f"\n📊 LAYER STATISTICS:")
        layer_types = {}
        for name, module in model.named_modules():
            layer_type = type(module).__name__
            if layer_type not in layer_types:
                layer_types[layer_type] = 0
            layer_types[layer_type] += 1
        
        for layer_type, count in sorted(layer_types.items()):
            print(f"  {layer_type}: {count}")
        
        return model, device
        
    except Exception as e:
        print(f"❌ Error analyzing model: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def check_model_output_range(model=None, device=None):
    """Check what range the model is outputting with detailed analysis"""
    print("\n" + "="*80)
    print("MODEL OUTPUT RANGE CHECK")
    print("="*80)
    
    if model is None or device is None:
        model, device = check_model_architecture()
        if model is None:
            return
    
    configs = Configs()
    
    dim = configs.dim if hasattr(configs, 'dim') else 32
    img1_dim = configs.c_dim * 2
    img2_dim = configs.c_dim * 2
    H = configs.patch_size[0] if hasattr(configs, 'patch_size') else 64
    W = configs.patch_size[1] if hasattr(configs, 'patch_size') else 64
    
    # Try to load checkpoint
    checkpoint_file = os.path.join(configs.checkpoint_dir, 'checkpoint.tar')
    checkpoint_loaded = False
    
    if os.path.isfile(checkpoint_file):
        try:
            checkpoint = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
            if 'best_loss' in checkpoint:
                print(f"   Best loss: {checkpoint['best_loss']:.6f}")
            if 'loss' in checkpoint:
                print(f"   Latest loss: {checkpoint['loss']:.6f}")
            checkpoint_loaded = True
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
    else:
        print(f"⚠️  No checkpoint found at {checkpoint_file}")
        print(f"   Testing with UNTRAINED model (outputs will be random)")
    
    model.eval()
    
    # Test with multiple input types
    test_cases = [
        ("Random normal (small)", lambda: torch.randn(1, img1_dim, H, W).to(device) * 0.1),
        ("Random normal (medium)", lambda: torch.randn(1, img1_dim, H, W).to(device) * 0.5),
        ("Random uniform [-1, 1]", lambda: torch.rand(1, img1_dim, H, W).to(device) * 2 - 1),
        ("Zeros", lambda: torch.zeros(1, img1_dim, H, W).to(device)),
        ("Ones", lambda: torch.ones(1, img1_dim, H, W).to(device)),
    ]
    
    print(f"\n🔬 TESTING MODEL WITH VARIOUS INPUTS:")
    print(f"   {'Input Type':<30} {'Min Output':<12} {'Max Output':<12} {'Mean':<12} {'Std':<12}")
    print(f"   {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
    
    outputs = []
    
    with torch.no_grad():
        for test_name, input_fn in test_cases:
            img1 = input_fn()
            img2 = input_fn()
            sum1 = input_fn()
            sum2 = input_fn()
            
            output = model(img1, img2, sum1, sum2)
            outputs.append((test_name, output))
            
            print(f"   {test_name:<30} {output.min().item():>11.6f} {output.max().item():>11.6f} {output.mean().item():>11.6f} {output.std().item():>11.6f}")
    
    # Analyze outputs
    print(f"\n⚠️  OUTPUT ANALYSIS:")
    issues = []
    
    for test_name, output in outputs:
        if abs(output.mean().item()) < 0.001 and output.std().item() < 0.01:
            issues.append(f"{test_name}: Output nearly zero (mean={output.mean().item():.6f}, std={output.std().item():.6f})")
        
        if output.std().item() < 0.001:
            issues.append(f"{test_name}: Output has collapsed (std={output.std().item():.6f})")
        
        if torch.isnan(output).any():
            issues.append(f"{test_name}: NaN values in output")
        
        if torch.isinf(output).any():
            issues.append(f"{test_name}: Infinite values in output")
    
    if not checkpoint_loaded:
        print(f"  ℹ️  Model not trained - outputs are random/uninitialized")
    elif issues:
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. ❌ {issue}")
    else:
        print(f"  ✅ Model outputs look reasonable")
    
    # Check gradient flow
    print(f"\n🌊 GRADIENT FLOW CHECK:")
    model.train()
    
    try:
        img1 = torch.randn(1, img1_dim, H, W).to(device) * 0.5
        img2 = torch.randn(1, img1_dim, H, W).to(device) * 0.5
        sum1 = torch.randn(1, img1_dim, H, W).to(device) * 0.5
        sum2 = torch.randn(1, img1_dim, H, W).to(device) * 0.5
        ref = torch.randn(1, 3, H, W).to(device) * 0.5
        
        img1.requires_grad = True
        
        output = model(img1, img2, sum1, sum2)
        criterion = HDRLoss()
        loss = criterion(tonemap(output), tonemap(ref))
        loss.backward()
        
        # Check gradients
        grad_stats = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_stats.append((name, grad_norm))
        
        if grad_stats:
            grad_stats.sort(key=lambda x: x[1], reverse=True)
            print(f"  Top 5 layers by gradient norm:")
            for name, norm in grad_stats[:5]:
                print(f"    {name}: {norm:.6f}")
            
            print(f"  Bottom 5 layers by gradient norm:")
            for name, norm in grad_stats[-5:]:
                print(f"    {name}: {norm:.6f}")
            
            # Check for vanishing/exploding gradients
            min_grad = min(g[1] for g in grad_stats)
            max_grad = max(g[1] for g in grad_stats)
            
            if min_grad < 1e-7:
                print(f"  ⚠️  Very small gradients detected (min={min_grad:.2e})")
            if max_grad > 100:
                print(f"  ⚠️  Very large gradients detected (max={max_grad:.2e})")
        else:
            print(f"  ❌ No gradients computed!")
            
    except Exception as e:
        print(f"  ❌ Error checking gradients: {e}")
    
    model.eval()


def check_training_data():
    """Analyze training data statistics"""
    print("\n" + "="*80)
    print("TRAINING DATA ANALYSIS")
    print("="*80)
    
    configs = Configs()
    
    # Look for data directory
    data_dir = configs.data_dir if hasattr(configs, 'data_dir') else './data'
    
    print(f"\n📁 Data directory: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"  ❌ Data directory not found")
        return
    
    # Look for training patches
    patch_dir = os.path.join(data_dir, 'train_patches')
    if os.path.exists(patch_dir):
        patch_files = glob(os.path.join(patch_dir, '*.pkl'))
        print(f"  Found {len(patch_files)} training patches")
        
        if patch_files:
            # Sample some patches
            sample_size = min(10, len(patch_files))
            print(f"\n  Analyzing {sample_size} random patches...")
            
            ref_hdr_stats = []
            in_ldr_stats = []
            
            import random
            sampled_files = random.sample(patch_files, sample_size)
            
            for pkl_file in sampled_files:
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    if 'ref_HDR' in data:
                        ref_hdr = data['ref_HDR']
                        ref_hdr_stats.append({
                            'min': ref_hdr.min(),
                            'max': ref_hdr.max(),
                            'mean': ref_hdr.mean(),
                            'std': ref_hdr.std()
                        })
                    
                    if 'in_LDR' in data:
                        in_ldr = data['in_LDR']
                        in_ldr_stats.append({
                            'min': in_ldr.min(),
                            'max': in_ldr.max(),
                            'mean': in_ldr.mean(),
                            'std': in_ldr.std()
                        })
                
                except Exception as e:
                    print(f"    Error reading {pkl_file}: {e}")
            
            if ref_hdr_stats:
                print(f"\n  📊 Reference HDR Statistics (from {len(ref_hdr_stats)} patches):")
                print(f"    Min:  {np.mean([s['min'] for s in ref_hdr_stats]):.6f} ± {np.std([s['min'] for s in ref_hdr_stats]):.6f}")
                print(f"    Max:  {np.mean([s['max'] for s in ref_hdr_stats]):.6f} ± {np.std([s['max'] for s in ref_hdr_stats]):.6f}")
                print(f"    Mean: {np.mean([s['mean'] for s in ref_hdr_stats]):.6f} ± {np.std([s['mean'] for s in ref_hdr_stats]):.6f}")
                print(f"    Std:  {np.mean([s['std'] for s in ref_hdr_stats]):.6f} ± {np.std([s['std'] for s in ref_hdr_stats]):.6f}")
            
            if in_ldr_stats:
                print(f"\n  📊 Input LDR Statistics (from {len(in_ldr_stats)} patches):")
                print(f"    Min:  {np.mean([s['min'] for s in in_ldr_stats]):.6f} ± {np.std([s['min'] for s in in_ldr_stats]):.6f}")
                print(f"    Max:  {np.mean([s['max'] for s in in_ldr_stats]):.6f} ± {np.std([s['max'] for s in in_ldr_stats]):.6f}")
                print(f"    Mean: {np.mean([s['mean'] for s in in_ldr_stats]):.6f} ± {np.std([s['mean'] for s in in_ldr_stats]):.6f}")
                print(f"    Std:  {np.mean([s['std'] for s in in_ldr_stats]):.6f} ± {np.std([s['std'] for s in in_ldr_stats]):.6f}")


def compare_with_reference():
    """Comprehensive comparison between model output and reference HDR"""
    print("\n" + "="*80)
    print("DETAILED COMPARISON WITH GROUND TRUTH")
    print("="*80)
    
    configs = Configs()
    sample_dir = configs.sample_dir if hasattr(configs, 'sample_dir') else './samples/u2net'
    
    if not os.path.exists(sample_dir):
        print(f"  ❌ Sample directory not found: {sample_dir}")
        return
    
    # Find output and reference
    subdirs = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
    
    if not subdirs:
        print(f"  ❌ No output directories found")
        return
    
    latest_dir = os.path.join(sample_dir, subdirs[0])
    output_hdr = os.path.join(latest_dir, 'hdr.hdr')
    ref_hdr = os.path.join(latest_dir, 'ref_hdr.hdr')
    
    if not os.path.exists(output_hdr):
        print(f"  ❌ Output HDR not found: {output_hdr}")
        return
    
    print(f"\n  📁 Files:")
    print(f"    Output:    {output_hdr}")
    
    output_img = cv2.imread(output_hdr, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if output_img is None:
        print(f"  ❌ Could not read output HDR")
        return
    output_img = output_img[..., ::-1]  # BGR to RGB
    
    if not os.path.exists(ref_hdr):
        print(f"    ⚠️  Reference HDR not found: {ref_hdr}")
        return
    
    print(f"    Reference: {ref_hdr}")
    ref_img = cv2.imread(ref_hdr, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
    if ref_img is None:
        print(f"  ❌ Could not read reference HDR")
        return
    ref_img = ref_img[..., ::-1]  # BGR to RGB
    
    # Check shape compatibility
    print(f"\n  📐 SHAPE INFORMATION:")
    print(f"    Output shape:    {output_img.shape}")
    print(f"    Reference shape: {ref_img.shape}")
    
    if output_img.shape != ref_img.shape:
        print(f"    ⚠️  Shape mismatch detected!")
        
        # Try to match shapes
        if output_img.shape[:2] != ref_img.shape[:2]:
            print(f"    Resizing output to match reference...")
            output_img_original = output_img.copy()
            output_img = cv2.resize(output_img, (ref_img.shape[1], ref_img.shape[0]), interpolation=cv2.INTER_LINEAR)
            print(f"    New output shape: {output_img.shape}")
        
        if output_img.shape[2] != ref_img.shape[2]:
            print(f"    ❌ Channel mismatch: output has {output_img.shape[2]} channels, reference has {ref_img.shape[2]}")
            print(f"    Cannot proceed with comparison")
            return
    else:
        print(f"    ✅ Shapes match")
    
    # Basic comparison
    print(f"\n  📊 BASIC STATISTICS COMPARISON:")
    print(f"    {'Metric':<20} {'Output':<15} {'Reference':<15} {'Difference':<15} {'Rel. Error':<15}")
    print(f"    {'-'*20} {'-'*15} {'-'*15} {'-'*15} {'-'*15}")
    
    metrics = [
        ('Min', np.min, output_img, ref_img),
        ('Max', np.max, output_img, ref_img),
        ('Mean', np.mean, output_img, ref_img),
        ('Median', np.median, output_img, ref_img),
        ('Std Dev', np.std, output_img, ref_img),
        ('10th percentile', lambda x: np.percentile(x, 10), output_img, ref_img),
        ('90th percentile', lambda x: np.percentile(x, 90), output_img, ref_img),
    ]
    
    for name, func, out, ref in metrics:
        out_val = func(out)
        ref_val = func(ref)
        diff = out_val - ref_val
        rel_err = (diff / (ref_val + 1e-8)) * 100 if ref_val != 0 else 0
        print(f"    {name:<20} {out_val:>14.6f} {ref_val:>14.6f} {diff:>14.6f} {rel_err:>13.2f}%")
    
    # Per-channel comparison
    print(f"\n  🎨 PER-CHANNEL COMPARISON:")
    for i, channel in enumerate(['R', 'G', 'B']):
        out_ch = output_img[:, :, i]
        ref_ch = ref_img[:, :, i]
        
        mse_ch = np.mean((out_ch - ref_ch) ** 2)
        mae_ch = np.mean(np.abs(out_ch - ref_ch))
        
        print(f"    {channel} channel:")
        print(f"      Output  - Mean: {out_ch.mean():>10.6f}  Std: {out_ch.std():>10.6f}  Range: [{out_ch.min():>10.6f}, {out_ch.max():>10.6f}]")
        print(f"      Ref     - Mean: {ref_ch.mean():>10.6f}  Std: {ref_ch.std():>10.6f}  Range: [{ref_ch.min():>10.6f}, {ref_ch.max():>10.6f}]")
        print(f"      MSE: {mse_ch:.6f}  MAE: {mae_ch:.6f}")
    
    # Error metrics
    print(f"\n  📈 ERROR METRICS (HDR Space):")
    
    # MSE
    mse = np.mean((output_img - ref_img) ** 2)
    print(f"    MSE (Mean Squared Error):       {mse:.8f}")
    
    # RMSE
    rmse = np.sqrt(mse)
    print(f"    RMSE (Root Mean Squared Error): {rmse:.8f}")
    
    # MAE
    mae = np.mean(np.abs(output_img - ref_img))
    print(f"    MAE (Mean Absolute Error):      {mae:.8f}")
    
    # PSNR
    if mse > 0:
        max_val = ref_img.max()
        psnr = 10 * np.log10(max_val ** 2 / mse)
        print(f"    PSNR (Peak Signal-to-Noise):    {psnr:.2f} dB")
    else:
        print(f"    PSNR (Peak Signal-to-Noise):    ∞ (perfect match)")
    
    # Relative error
    rel_error = np.mean(np.abs(output_img - ref_img) / (np.abs(ref_img) + 1e-8)) * 100
    print(f"    Mean Relative Error:            {rel_error:.2f}%")
    
    # Structural similarity (simple version)
    correlation = np.corrcoef(output_img.flatten(), ref_img.flatten())[0, 1]
    print(f"    Correlation Coefficient:        {correlation:.4f}")
    
    # Compute loss using actual loss function
    print(f"\n  🎯 ACTUAL TRAINING LOSS:")
    try:
        # Convert to torch tensors in [-1, 1] range
        output_tensor = torch.from_numpy(output_img * 2 - 1).permute(2, 0, 1).unsqueeze(0).float()
        ref_tensor = torch.from_numpy(ref_img * 2 - 1).permute(2, 0, 1).unsqueeze(0).float()
        
        criterion = HDRLoss()
        
        # Compute loss in tone-mapped space (as done in training)
        loss_value = criterion(tonemap(output_tensor), tonemap(ref_tensor))
        print(f"    Training Loss (tone-mapped):    {loss_value.item():.8f}")
        
        # Also compute loss in linear space
        loss_linear = torch.nn.functional.mse_loss(output_tensor, ref_tensor)
        print(f"    MSE Loss (linear space):        {loss_linear.item():.8f}")
        
    except Exception as e:
        print(f"    ❌ Error computing training loss: {e}")
    
    # Tone-mapped comparison
    print(f"\n  🖼️  TONE-MAPPED COMPARISON (what humans see):")
    
    try:
        # Ensure shapes match before tone mapping
        if output_img.shape != ref_img.shape:
            print(f"    ⚠️  Shapes still don't match, skipping tone-mapped comparison")
        else:
            # Convert to [-1, 1] for tone mapping
            output_tonemapped = tonemap_np(output_img * 2 - 1)
            ref_tonemapped = tonemap_np(ref_img * 2 - 1)
            
            # Convert to [0, 1]
            output_tonemapped = (output_tonemapped + 1) / 2
            ref_tonemapped = (ref_tonemapped + 1) / 2
            
            # MSE in tone-mapped space
            mse_tone = np.mean((output_tonemapped - ref_tonemapped) ** 2)
            print(f"    MSE (tone-mapped):              {mse_tone:.8f}")
            
            # PSNR in tone-mapped space
            if mse_tone > 0:
                psnr_tone = 10 * np.log10(1.0 / mse_tone)
                print(f"    PSNR (tone-mapped):             {psnr_tone:.2f} dB")
            
            # Save tone-mapped comparison
            output_8bit = (np.clip(output_tonemapped, 0, 1) * 255).astype(np.uint8)
            ref_8bit = (np.clip(ref_tonemapped, 0, 1) * 255).astype(np.uint8)
            
            cv2.imwrite('debug_output/output_tonemapped.png', output_8bit[..., ::-1])
            cv2.imwrite('debug_output/reference_tonemapped.png', ref_8bit[..., ::-1])
            print(f"\n    ✅ Saved tone-mapped output: debug_output/output_tonemapped.png")
            print(f"    ✅ Saved tone-mapped reference: debug_output/reference_tonemapped.png")
        
    except Exception as e:
        print(f"    ❌ Error computing tone-mapped comparison: {e}")
        import traceback
        traceback.print_exc()
    
    # Error distribution analysis
    print(f"\n  📊 ERROR DISTRIBUTION:")
    
    error = np.abs(output_img - ref_img)
    
    print(f"    Error percentiles:")
    for p in [50, 75, 90, 95, 99]:
        print(f"      {p}th percentile: {np.percentile(error, p):.6f}")
    
    # Count pixels by error magnitude
    total_pixels = error.size
    error_ranges = [
        ("< 0.001", error < 0.001),
        ("0.001-0.01", (error >= 0.001) & (error < 0.01)),
        ("0.01-0.1", (error >= 0.01) & (error < 0.1)),
        ("0.1-1.0", (error >= 0.1) & (error < 1.0)),
        (">= 1.0", error >= 1.0),
    ]
    
    print(f"\n    Pixels by absolute error:")
    for label, mask in error_ranges:
        count = mask.sum()
        print(f"      {label:>12}: {count:>10} ({100*count/total_pixels:>6.2f}%)")
    
    # Visual comparison
    print(f"\n  🎨 CREATING VISUAL COMPARISONS...")
    
    # Side-by-side comparison (tone-mapped)
    try:
        if output_img.shape == ref_img.shape:
            # Convert to tone-mapped for visualization
            output_tonemapped = tonemap_np(output_img * 2 - 1)
            ref_tonemapped = tonemap_np(ref_img * 2 - 1)
            output_tonemapped = (output_tonemapped + 1) / 2
            ref_tonemapped = (ref_tonemapped + 1) / 2
            
            output_8bit = (np.clip(output_tonemapped, 0, 1) * 255).astype(np.uint8)
            ref_8bit = (np.clip(ref_tonemapped, 0, 1) * 255).astype(np.uint8)
            
            h, w = output_img.shape[:2]
            
            # Create side-by-side
            comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            comparison[:, :w] = output_8bit
            comparison[:, w:] = ref_8bit
            
            # Add labels
            cv2.putText(comparison, 'Output', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, 'Reference', (w+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite('debug_output/side_by_side.png', comparison[..., ::-1])
            print(f"    ✅ Saved side-by-side: debug_output/side_by_side.png")
        else:
            print(f"    ⚠️  Cannot create side-by-side due to shape mismatch")
    except Exception as e:
        print(f"    ⚠️  Could not create side-by-side: {e}")
    
    # Difference maps (only if shapes match)
    if output_img.shape == ref_img.shape:
        try:
            diff = np.abs(output_img - ref_img)
            
            # Linear difference
            diff_normalized = np.clip(diff / (diff.max() + 1e-8), 0, 1)
            cv2.imwrite('debug_output/difference_linear.png', (diff_normalized * 255).astype(np.uint8)[..., ::-1])
            print(f"    ✅ Saved linear difference: debug_output/difference_linear.png")
            
            # Log difference (shows small errors better)
            diff_log = np.log1p(diff * 100) / np.log1p(100)
            diff_log = np.clip(diff_log, 0, 1)
            cv2.imwrite('debug_output/difference_log.png', (diff_log * 255).astype(np.uint8)[..., ::-1])
            print(f"    ✅ Saved log difference: debug_output/difference_log.png")
            
            # Heat map (colored)
            try:
                diff_gray = np.mean(diff, axis=2)
                diff_gray = np.clip(diff_gray / (np.percentile(diff_gray, 99) + 1e-8), 0, 1)
                diff_colored = cv2.applyColorMap((diff_gray * 255).astype(np.uint8), cv2.COLORMAP_JET)
                cv2.imwrite('debug_output/difference_heatmap.png', diff_colored)
                print(f"    ✅ Saved heat map: debug_output/difference_heatmap.png")
            except Exception as e:
                print(f"    ⚠️  Could not create heat map: {e}")
        except Exception as e:
            print(f"    ⚠️  Could not create difference maps: {e}")
    else:
        print(f"    ⚠️  Skipping difference maps due to shape mismatch")
    
    # Histogram comparison
    try:
        if output_img.shape == ref_img.shape:
            # Convert to tone-mapped for histogram
            output_tonemapped = tonemap_np(output_img * 2 - 1)
            ref_tonemapped = tonemap_np(ref_img * 2 - 1)
            output_tonemapped = (output_tonemapped + 1) / 2
            ref_tonemapped = (ref_tonemapped + 1) / 2
            
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.hist(output_img.flatten(), bins=100, alpha=0.7, label='Output', color='blue')
            plt.hist(ref_img.flatten(), bins=100, alpha=0.7, label='Reference', color='red')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('HDR Value Distribution')
            plt.legend()
            plt.yscale('log')
            
            plt.subplot(1, 3, 2)
            plt.hist(output_tonemapped.flatten(), bins=100, alpha=0.7, label='Output', color='blue')
            plt.hist(ref_tonemapped.flatten(), bins=100, alpha=0.7, label='Reference', color='red')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Tone-mapped Distribution')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            diff = np.abs(output_img - ref_img)
            plt.hist(diff.flatten(), bins=100, alpha=0.7, color='orange')
            plt.xlabel('Absolute Error')
            plt.ylabel('Frequency')
            plt.title('Error Distribution')
            plt.yscale('log')
            
            plt.tight_layout()
            plt.savefig('debug_output/histogram_comparison.png', dpi=150)
            plt.close()
            print(f"    ✅ Saved histogram comparison: debug_output/histogram_comparison.png")
        else:
            print(f"    ⚠️  Cannot create histogram comparison due to shape mismatch")
    except Exception as e:
        print(f"    ⚠️  Could not create histogram comparison: {e}")
        import traceback
        traceback.print_exc()
    
    # Issue detection
    print(f"\n  ⚠️  COMPARISON ISSUES:")
    issues = []
    
    # Check for shape mismatch
    if output_img.shape != ref_img.shape:
        issues.append(f"Shape mismatch: output {output_img.shape} vs reference {ref_img.shape}")
        issues.append("This may indicate patch processing issues or incorrect saving")
    
    if abs(output_img.mean() - ref_img.mean()) / (ref_img.mean() + 1e-8) > 0.5:
        issues.append(f"Large mean difference: output={output_img.mean():.6f} vs ref={ref_img.mean():.6f}")
    
    if output_img.max() < ref_img.max() * 0.1:
        issues.append(f"Output max is only {100*output_img.max()/ref_img.max():.1f}% of reference max")
    
    if mse > 0.01:
        issues.append(f"High MSE: {mse:.6f} (target should be < 0.01)")
    
    if rel_error > 50:
        issues.append(f"High relative error: {rel_error:.1f}% (target should be < 20%)")
    
    if correlation < 0.5:
        issues.append(f"Poor correlation: {correlation:.4f} (should be > 0.8)")
    
    # Check if output is uniformly scaled
    if output_img.std() > 0 and ref_img.std() > 0:
        scale_ratio = (output_img.std() / ref_img.std())
        if scale_ratio < 0.1 or scale_ratio > 10:
            issues.append(f"Large scale mismatch: output std / ref std = {scale_ratio:.2f}")
    
    if issues:
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. ❌ {issue}")
    else:
        print(f"    ✅ No major issues detected in comparison")
    
    # Recommendations
    print(f"\n  💡 RECOMMENDATIONS:")
    
    if output_img.shape != ref_img.shape:
        print(f"    • Shape mismatch between output and reference")
        print(f"      → Check patch processing in eval_one_epoch")
        print(f"      → Verify Hann window blending is correctly accumulating patches")
        print(f"      → Check if dump_sample is saving with correct dimensions")
    
    if output_img.mean() < ref_img.mean() * 0.1:
        print(f"    • Output is much darker than reference")
        print(f"      → Check if model is learning (look at training loss)")
        print(f"      → Verify data preprocessing/normalization")
        print(f"      → Check if tonemap function is applied correctly during training")
    
    if mse > 0.1:
        print(f"    • High MSE indicates poor reconstruction")
        print(f"      → Model needs more training")
        print(f"      → Check loss function is appropriate")
        print(f"      → Verify model architecture")
    
    if correlation < 0.5:
        print(f"    • Low correlation suggests output doesn't match reference structure")
        print(f"      → Model may not be learning the right features")
        print(f"      → Check training data quality")
        print(f"      → Verify model is not collapsed")


def check_multiple_samples():
    """Check multiple output samples for consistency"""
    print("\n" + "="*80)
    print("MULTIPLE SAMPLES ANALYSIS")
    print("="*80)
    
    configs = Configs()
    sample_dir = configs.sample_dir if hasattr(configs, 'sample_dir') else './samples/u2net'
    
    if not os.path.exists(sample_dir):
        print(f"  ⚠️  Sample directory doesn't exist")
        return
    
    subdirs = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
    
    if not subdirs:
        print(f"  ⚠️  No output directories found")
        return
    
    print(f"\n  Found {len(subdirs)} output samples")
    
    # Analyze up to 10 samples
    samples_to_check = min(10, len(subdirs))
    print(f"  Analyzing {samples_to_check} samples for consistency...\n")
    
    output_stats = []
    ref_stats = []
    
    for i, subdir in enumerate(subdirs[:samples_to_check]):
        sample_path = os.path.join(sample_dir, subdir)
        output_file = os.path.join(sample_path, 'hdr.hdr')
        ref_file = os.path.join(sample_path, 'ref_hdr.hdr')
        
        if os.path.exists(output_file):
            output_img = cv2.imread(output_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if output_img is not None:
                output_img = output_img[..., ::-1]
                output_stats.append({
                    'name': subdir,
                    'min': output_img.min(),
                    'max': output_img.max(),
                    'mean': output_img.mean(),
                    'std': output_img.std(),
                })
        
        if os.path.exists(ref_file):
            ref_img = cv2.imread(ref_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)
            if ref_img is not None:
                ref_img = ref_img[..., ::-1]
                ref_stats.append({
                    'name': subdir,
                    'min': ref_img.min(),
                    'max': ref_img.max(),
                    'mean': ref_img.mean(),
                    'std': ref_img.std(),
                })
    
    if output_stats:
        print(f"  📊 OUTPUT STATISTICS ACROSS {len(output_stats)} SAMPLES:")
        print(f"    {'Metric':<15} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std Dev':<12}")
        print(f"    {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for metric in ['min', 'max', 'mean', 'std']:
            values = [s[metric] for s in output_stats]
            print(f"    {metric.capitalize():<15} {min(values):>11.6f} {max(values):>11.6f} {np.mean(values):>11.6f} {np.std(values):>11.6f}")
        
        # Check consistency
        print(f"\n  ⚠️  CONSISTENCY CHECK:")
        
        issues = []
        mean_values = [s['mean'] for s in output_stats]
        mean_std = np.std(mean_values)
        
        if mean_std / (np.mean(mean_values) + 1e-8) > 0.5:
            issues.append(f"High variance in output means across samples (std={mean_std:.6f})")
        
        # Check if all outputs are similarly dark/bright
        all_dark = all(s['mean'] < 0.01 for s in output_stats)
        all_bright = all(s['mean'] > 0.5 for s in output_stats)
        
        if all_dark:
            issues.append("All outputs are consistently dark - systematic issue")
        if all_bright:
            issues.append("All outputs are consistently bright - may be overexposed")
        
        if issues:
            for i, issue in enumerate(issues, 1):
                print(f"    {i}. ❌ {issue}")
        else:
            print(f"    ✅ Outputs are consistent across samples")
    
    if ref_stats:
        print(f"\n  📊 REFERENCE STATISTICS ACROSS {len(ref_stats)} SAMPLES:")
        print(f"    {'Metric':<15} {'Min':<12} {'Max':<12} {'Mean':<12} {'Std Dev':<12}")
        print(f"    {'-'*15} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")
        
        for metric in ['min', 'max', 'mean', 'std']:
            values = [s[metric] for s in ref_stats]
            print(f"    {metric.capitalize():<15} {min(values):>11.6f} {max(values):>11.6f} {np.mean(values):>11.6f} {np.std(values):>11.6f}")


def print_summary_report():
    """Print a summary report with key findings"""
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY REPORT")
    print("="*80)
    
    print("\n📁 Generated Files (check debug_output/ folder):")
    print("  Visual Comparisons:")
    print("    • side_by_side.png          - Output vs Reference side-by-side")
    print("    • output_tonemapped.png     - Model output (tone-mapped)")
    print("    • reference_tonemapped.png  - Ground truth (tone-mapped)")
    print("\n  Difference Maps:")
    print("    • difference_linear.png     - Absolute difference (linear)")
    print("    • difference_log.png        - Absolute difference (log scale)")
    print("    • difference_heatmap.png    - Error heat map (colored)")
    print("\n  Distributions:")
    print("    • histogram.png             - HDR value distribution")
    print("    • histogram_log.png         - HDR value distribution (log)")
    print("    • histogram_comparison.png  - Output vs Reference histograms")
    print("\n  Tone Mapping Methods:")
    print("    • tonemap_simple.png        - Simple gamma correction")
    print("    • tonemap_log.png           - Logarithmic tone mapping")
    print("    • tonemap_mu.png            - Mu-law compression (training method)")
    print("    • tonemap_auto.png          - Auto-exposure")
    print("    • tonemap_reinhard.png      - Reinhard tone mapping")
    print("    • tonemap_stretch.png       - Linear stretch")
    
    print("\n🔍 Key Things to Check:")
    print("  1. MODEL STATUS:")
    print("     □ Is model trained? (check checkpoint epoch number)")
    print("     □ Is training loss decreasing?")
    print("     □ Are gradients flowing? (not too small/large)")
    print("\n  2. OUTPUT QUALITY:")
    print("     □ Is output range reasonable? (not all zeros/near-zero)")
    print("     □ Is output structure similar to reference?")
    print("     □ Are there NaN/Inf values?")
    print("\n  3. DATA PIPELINE:")
    print("     □ Is training data normalized correctly?")
    print("     □ Are exposure values applied properly?")
    print("     □ Does tonemap function match training?")
    print("\n  4. ARCHITECTURE:")
    print("     □ Is output layer activation appropriate? (no ReLU/Sigmoid)")
    print("     □ Is model capacity sufficient?")
    print("     □ Are there any collapsed layers?")
    

    
def check_training_progress():
    """Analyze training progress from logs and checkpoints"""
    print("\n" + "="*80)
    print("TRAINING PROGRESS ANALYSIS")
    print("="*80)
    
    configs = Configs()
    log_dir = os.path.join(configs.checkpoint_dir, 'logs')
    
    # Check for training stats
    stats_file = os.path.join(log_dir, 'training_stats.json')
    if os.path.exists(stats_file):
        try:
            with open(stats_file, 'r') as f:
                stats = json.load(f)
            
            if stats:
                print(f"\n  📊 TRAINING STATISTICS ({len(stats)} epochs recorded):")
                
                # Get loss history
                train_losses = [s['train_loss'] for s in stats if 'train_loss' in s]
                eval_losses = [s['eval_loss'] for s in stats if 'eval_loss' in s and s['eval_loss'] > 0]
                
                if train_losses:
                    print(f"\n  Training Loss:")
                    print(f"    First epoch:  {train_losses[0]:.6f}")
                    print(f"    Latest epoch: {train_losses[-1]:.6f}")
                    print(f"    Best:         {min(train_losses):.6f}")
                    print(f"    Change:       {train_losses[-1] - train_losses[0]:.6f}")
                    
                    # Check convergence
                    if len(train_losses) > 5:
                        recent_avg = np.mean(train_losses[-5:])
                        early_avg = np.mean(train_losses[:5])
                        improvement = (early_avg - recent_avg) / early_avg * 100
                        
                        print(f"    Improvement:  {improvement:.1f}%")
                        
                        if improvement < 5:
                            print(f"    ⚠️  Training may have plateaued")
                        elif improvement > 50:
                            print(f"    ✅ Good training progress")
                
                if eval_losses:
                    print(f"\n  Validation Loss:")
                    print(f"    First:  {eval_losses[0]:.6f}")
                    print(f"    Latest: {eval_losses[-1]:.6f}")
                    print(f"    Best:   {min(eval_losses):.6f}")
                
                # Plot training curve
                try:
                    plt.figure(figsize=(10, 6))
                    epochs = [s['epoch'] for s in stats]
                    
                    if train_losses:
                        plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
                    
                    if eval_losses:
                        eval_epochs = [s['epoch'] for s in stats if 'eval_loss' in s and s['eval_loss'] > 0]
                        plt.plot(eval_epochs, eval_losses, 'r-', label='Validation Loss', linewidth=2)
                    
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Training Progress')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.yscale('log')
                    plt.tight_layout()
                    plt.savefig('debug_output/training_curve.png', dpi=150)
                    plt.close()
                    print(f"\n  ✅ Saved training curve: debug_output/training_curve.png")
                except Exception as e:
                    print(f"  ⚠️  Could not plot training curve: {e}")
                
                # Learning rate history
                lrs = [s['learning_rate'] for s in stats if 'learning_rate' in s]
                if lrs:
                    print(f"\n  Learning Rate:")
                    print(f"    Initial: {lrs[0]:.8f}")
                    print(f"    Current: {lrs[-1]:.8f}")
                
                # Time statistics
                if 'epoch_time' in stats[0]:
                    times = [s['epoch_time'] for s in stats if 'epoch_time' in s]
                    print(f"\n  Time per Epoch:")
                    print(f"    Average: {np.mean(times):.2f}s")
                    print(f"    Total:   {sum(times)/3600:.2f} hours")
        
        except Exception as e:
            print(f"  ❌ Error reading training stats: {e}")
    else:
        print(f"  ⚠️  No training stats file found: {stats_file}")
    
    # Check checkpoint
    checkpoint_file = os.path.join(configs.checkpoint_dir, 'checkpoint.tar')
    if os.path.exists(checkpoint_file):
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            print(f"\n  📦 CHECKPOINT INFO:")
            print(f"    Epoch:     {checkpoint.get('epoch', 'N/A')}")
            print(f"    Loss:      {checkpoint.get('loss', 'N/A')}")
            print(f"    Best loss: {checkpoint.get('best_loss', 'N/A')}")
            
            if 'config' in checkpoint:
                config = checkpoint['config']
                print(f"\n  Model Config:")
                for key, value in config.items():
                    print(f"    {key}: {value}")
        
        except Exception as e:
            print(f"  ⚠️  Could not read checkpoint: {e}")
    else:
        print(f"  ⚠️  No checkpoint found: {checkpoint_file}")


def main():
    """Run comprehensive diagnostics"""
    print("\n" + "="*80)
    print("HDR OUTPUT QUALITY COMPREHENSIVE DIAGNOSTIC")
    print("="*80)
    print("\nThis script will analyze your HDR output in detail to identify issues.")
    print("All visualizations will be saved to: debug_output/")
    
    os.makedirs('debug_output', exist_ok=True)
    
    # 0. Check training progress
    print("\n" + "="*80)
    print("STEP 0: TRAINING PROGRESS")
    print("="*80)
    check_training_progress()
    
    # 1. Check model architecture
    print("\n" + "="*80)
    print("STEP 1: MODEL ARCHITECTURE ANALYSIS")
    print("="*80)
    model, device = check_model_architecture()
    
    # 2. Check model outputs
    print("\n" + "="*80)
    print("STEP 2: MODEL OUTPUT RANGE TEST")
    print("="*80)
    check_model_output_range(model, device)
    
    # 3. Check training data
    print("\n" + "="*80)
    print("STEP 3: TRAINING DATA ANALYSIS")
    print("="*80)
    check_training_data()
    
    # 4. Check multiple samples
    print("\n" + "="*80)
    print("STEP 4: MULTIPLE SAMPLES CHECK")
    print("="*80)
    check_multiple_samples()
    
    # 5. Analyze actual outputs
    print("\n" + "="*80)
    print("STEP 5: DETAILED OUTPUT ANALYSIS")
    print("="*80)
    
    configs = Configs()
    sample_dir = configs.sample_dir if hasattr(configs, 'sample_dir') else './samples/u2net'
    
    if os.path.exists(sample_dir):
        subdirs = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
        if subdirs:
            # Check both output and reference
            latest_dir = os.path.join(sample_dir, subdirs[0])
            
            print(f"\n  Analyzing latest sample: {subdirs[0]}")
            
            for filename in ['hdr.hdr', 'ref_hdr.hdr']:
                hdr_file = os.path.join(latest_dir, filename)
                if os.path.exists(hdr_file):
                    analyze_hdr_output(hdr_file)
        else:
            print(f"  ⚠️  No output directories in {sample_dir}")
    else:
        print(f"  ⚠️  Sample directory doesn't exist: {sample_dir}")
    
    # 6. Compare output with reference (ENHANCED)
    print("\n" + "="*80)
    print("STEP 6: GROUND TRUTH COMPARISON")
    print("="*80)
    compare_with_reference()
    
    # 7. Print summary report
    print_summary_report()
    
    print("\n" + "="*80)
    print("✅ DIAGNOSIS COMPLETE")
    print("="*80)
    print(f"\nAll visualization files saved to: debug_output/")
    print(f"Review the summary above and check the generated images.")
    print(f"\nTo re-run: python {__file__}\n")


if __name__ == "__main__":
    main()