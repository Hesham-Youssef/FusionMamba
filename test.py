import os
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.tools import *
from model.u2net import U2Net
import numpy as np
from config import Configs
from tqdm import tqdm
from pathlib import Path
from dataset_diag import HDRLoss
from dataset_diag import comprehensive_diagnostics, save_diagnostic_images


# Import the dataset with shuffling support
from utils.hdr_load_train_data import (
    U2NetTestDataset, 
    unshuffle_output
)


def make_blend_window(h, w, device, blend_type='gaussian'):
    """Create blending window for tiled inference"""
    if blend_type == 'gaussian':
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        dist = torch.sqrt(xx**2 + yy**2)
        sigma = 0.5
        weight = torch.exp(-(dist**2) / (2 * sigma**2))
        weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        return weight[None, None]
        
    elif blend_type == 'hann':
        wy = torch.hann_window(h, periodic=False, device=device)
        wx = torch.hann_window(w, periodic=False, device=device)
        w2d = wy[:, None] * wx[None, :]
        return w2d[None, None]


def process_image_tiled(model, img1, img2, sum1, sum2, patch_h, patch_w, device):
    """
    Process a full-resolution image using tiled inference with reflection padding.
    Matches the eval_one_epoch implementation.
    """
    stride = patch_h // 2
    PATCH_CHUNK = 60
    pad_size = stride // 2

    _, _, H_tensor, W_tensor = img1.shape

    if H_tensor > patch_h or W_tensor > patch_w:
        # Pad with reflection for clean borders
        img1_pad = F.pad(img1, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        img2_pad = F.pad(img2, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        _, _, H_pad, W_pad = img1_pad.shape
        
        # Generate patch coordinates
        h_starts = [0]
        h = stride
        while h < H_pad - patch_h:
            h_starts.append(h)
            h += stride
        if h_starts[-1] != H_pad - patch_h:
            h_starts.append(H_pad - patch_h)

        w_starts = [0]
        w = stride
        while w < W_pad - patch_w:
            w_starts.append(w)
            w += stride
        if w_starts[-1] != W_pad - patch_w:
            w_starts.append(W_pad - patch_w)

        patch_coords = [(hs, ws) for hs in h_starts for ws in w_starts]

        # Initialize result accumulators
        result_pad = torch.zeros((1, img1.shape[1], H_pad, W_pad), 
                                device=device, dtype=torch.float32)
        weight_map = torch.zeros((1, 1, H_pad, W_pad), device=device)

        # Create blending window
        blend = make_blend_window(patch_h, patch_w, device, blend_type='hann')

        # Process patches in chunks
        with torch.no_grad():
            for start in range(0, len(patch_coords), PATCH_CHUNK):
                batch = patch_coords[start:start + PATCH_CHUNK]
                batch_size = len(batch)

                b_img1, b_img2 = [], []
                for hs, ws in batch:
                    he = hs + patch_h
                    we = ws + patch_w
                    b_img1.append(img1_pad[:, :, hs:he, ws:we])
                    b_img2.append(img2_pad[:, :, hs:he, ws:we])

                b_img1 = torch.cat(b_img1, dim=0)
                b_img2 = torch.cat(b_img2, dim=0)
                b_sum1 = sum1.repeat(batch_size, 1, 1, 1)
                b_sum2 = sum2.repeat(batch_size, 1, 1, 1)

                # Forward pass
                out = model(b_img1, b_img2, b_sum1, b_sum2)
                out = out * blend

                # Accumulate results
                for i, (hs, ws) in enumerate(batch):
                    he = hs + patch_h
                    we = ws + patch_w
                    result_pad[:, :, hs:he, ws:we] += out[i:i+1]
                    weight_map[:, :, hs:he, ws:we] += blend

        # Normalize and remove padding
        result_pad = result_pad / torch.clamp(weight_map, min=1e-3)
        result = result_pad[:, :, pad_size:-pad_size, pad_size:-pad_size]
    else:
        # Small image - process with padding but no tiling
        img1_pad = F.pad(img1, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        img2_pad = F.pad(img2, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
        
        with torch.no_grad():
            result_pad = model(img1_pad, img2_pad, sum1, sum2)
            result = result_pad[:, :, pad_size:-pad_size, pad_size:-pad_size]

    return result


def save_comparison_image(output, reference, save_path, label1='Output', label2='Reference'):
    """Save side-by-side comparison image"""
    # Convert to numpy
    output_np = output.detach().cpu().numpy()[0]
    ref_np = reference.detach().cpu().numpy()[0]
    
    # Normalize to [0, 1]
    output_01 = np.clip((output_np + 1) / 2, 0, 1)
    ref_01 = np.clip((ref_np + 1) / 2, 0, 1)
    
    # Transpose to (H, W, C)
    output_01 = np.transpose(output_01, (1, 2, 0))
    ref_01 = np.transpose(ref_01, (1, 2, 0))
    
    # Convert to 8-bit
    output_8bit = (output_01 * 255).astype(np.uint8)
    ref_8bit = (ref_01 * 255).astype(np.uint8)
    
    # Create side-by-side comparison
    h, w = output_8bit.shape[:2]
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = output_8bit
    comparison[:, w:] = ref_8bit
    
    # Add labels
    cv2.putText(comparison, label1, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(comparison, label2, (w+10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save (OpenCV uses BGR)
    cv2.imwrite(save_path, comparison[..., ::-1])


def load_all_exposures(cache, scene_name, num_shots, H, W):
    """Load all exposure images for a scene"""
    from utils.hdr_load_train_data import load_memmap
    
    exposures = []
    summaries = []
    
    # Load summary shape
    sum_shape = np.load(cache.sum_meta_path(scene_name)).astype(int)
    sum_shape = (int(sum_shape[0]), int(sum_shape[1]), int(sum_shape[2]))
    
    for idx in range(num_shots):
        # Load image
        img = load_memmap(cache.img_path(scene_name, idx), (H, W, 3))
        img = torch.from_numpy(img.copy()).permute(2, 0, 1).contiguous()
        
        # Load summary
        summary = load_memmap(cache.sum_path(scene_name, idx), sum_shape)
        summary = torch.from_numpy(summary.copy()).permute(2, 0, 1).contiguous()
        
        exposures.append(img)
        summaries.append(summary)
    
    return exposures, summaries


def test_mode_individual(model, test_dataloader, configs, device, output_dir):
    """
    Mode A: Merge each exposure (including reference) with reference individually.
    For each exposure i in [0, num_shots-1], merge exposure_i with reference.
    """
    print("\n" + "="*80)
    print("TEST MODE: INDIVIDUAL MERGING")
    print("Each exposure is merged with the reference independently")
    print("="*80)
    
    model.eval()
    results = []
    
    patch_h, patch_w = configs.patch_size
    
    # Create output directory
    mode_dir = Path(output_dir) / "mode_individual"
    mode_dir.mkdir(parents=True, exist_ok=True)
    
    for batch_idx, data in enumerate(tqdm(test_dataloader, desc="Testing (Individual)")):
        sample_path, img1, img2, sum1, sum2, ref_HDR, unshuffle_indices, (H, W) = data
        sample_path = sample_path[0]
        scene_name = Path(sample_path).name
        
        # Move to device
        ref_HDR = ref_HDR.to(device)
        H_orig = H[0].item()
        W_orig = W[0].item()
        
        # Create scene output directory
        scene_dir = mode_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all exposures
        from utils.hdr_load_train_data import SceneCache
        cache = SceneCache("cache", enable_shuffle=configs.enable_shuffle)
        exposures, summaries = load_all_exposures(cache, scene_name, configs.num_shots, H_orig, W_orig)
        
        scene_results = []
        
        # Test each exposure with reference
        for exp_idx in range(configs.num_shots):
            exp_img = exposures[exp_idx].unsqueeze(0).to(device)
            exp_sum = summaries[exp_idx].unsqueeze(0).to(device)
            
            # Get reference (middle exposure)
            ref_idx = configs.num_shots // 2
            ref_img = exposures[ref_idx].unsqueeze(0).to(device)
            ref_sum = summaries[ref_idx].unsqueeze(0).to(device)
            
            # Process with tiled inference
            result = process_image_tiled(
                model, ref_img, exp_img, ref_sum, exp_sum, 
                patch_h, patch_w, device
            )
            
            # Unshuffle if needed
            if configs.enable_shuffle and unshuffle_indices.numel() > 0:
                unshuffle_np = unshuffle_indices[0].cpu().numpy()
                result_unshuffled = unshuffle_output(result, unshuffle_np, H_orig, W_orig)
                ref_unshuffled = unshuffle_output(ref_HDR, unshuffle_np, H_orig, W_orig)
            else:
                result_unshuffled = result
                ref_unshuffled = ref_HDR
            
            # Calculate loss
            loss, loss_dict = criterion(result_unshuffled, ref_unshuffled)
            scene_results.append({
                'exposure_idx': exp_idx,
                'loss': loss.item(),
                'loss_dict': {k: v for k, v in loss_dict.items()}
            })
            
            diagnostic_name = f"Individual - Scene {scene_name}, Exp {exp_idx}"
            comprehensive_diagnostics(
                result_unshuffled, ref_unshuffled,
                exp_img, ref_img, exp_sum, ref_sum,
                name=diagnostic_name
            )
            
            # Save outputs - dump_sample only takes 2 arguments
            output_filename = f"exp_{exp_idx:02d}_merged"
            exp_output_dir = scene_dir / output_filename
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            dump_sample(str(exp_output_dir), result_unshuffled.cpu().numpy())
            
            save_diagnostic_images(
                result_unshuffled, ref_unshuffled,
                exp_img, ref_img,
                str(exp_output_dir),
                idx=exp_idx
            )
            
            # Save comparison image (in parent scene directory to avoid conflicts)
            comparison_path = scene_dir / f"comparison_exp_{exp_idx:02d}.png"
            save_comparison_image(
                result_unshuffled, ref_unshuffled, 
                str(comparison_path),
                label1=f'Exp {exp_idx} + Ref',
                label2='Ground Truth'
            )
            
            # Format loss components for display
            loss_components = ', '.join([f'{k}={v:.4f}' for k, v in loss_dict.items()])
            print(f"  Scene {scene_name} | Exp {exp_idx} → Loss: {loss.item():.6f} ({loss_components})")
        
        # Save scene summary
        summary_path = scene_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Scene: {scene_name}\n")
            f.write(f"Mode: Individual Merging\n")
            f.write("="*60 + "\n")
            for res in scene_results:
                f.write(f"Exposure {res['exposure_idx']}: Total Loss = {res['loss']:.6f}\n")
                # Write loss components if available
                if 'loss_dict' in res:
                    for key, val in res['loss_dict'].items():
                        f.write(f"  {key}: {val:.6f}\n")
            avg_loss = np.mean([r['loss'] for r in scene_results])
            f.write("="*60 + "\n")
            f.write(f"Average Loss: {avg_loss:.6f}\n")
        
        results.append({
            'scene': scene_name,
            'results': scene_results
        })
    
    return results


def test_mode_iterative(model, test_dataloader, configs, device, output_dir):
    """
    Mode B: Iterative merging.
    Start with reference, iteratively merge with each exposure:
    ref → merge(exp0, ref) → merge(exp1, output) → merge(exp2, output) → ...
    """
    print("\n" + "="*80)
    print("TEST MODE: ITERATIVE MERGING")
    print("Start with reference, iteratively merge each exposure")
    print("="*80)
    
    model.eval()
    results = []
    
    patch_h, patch_w = configs.patch_size
    
    # Create output directory
    mode_dir = Path(output_dir) / "mode_iterative"
    mode_dir.mkdir(parents=True, exist_ok=True)
    
    for batch_idx, data in enumerate(tqdm(test_dataloader, desc="Testing (Iterative)")):
        sample_path, img1, img2, sum1, sum2, ref_HDR, unshuffle_indices, (H, W) = data
        sample_path = sample_path[0]
        scene_name = Path(sample_path).name
        
        # Move to device
        ref_HDR = ref_HDR.to(device)
        H_orig = H[0].item()
        W_orig = W[0].item()
        
        # Create scene output directory
        scene_dir = mode_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all exposures
        from utils.hdr_load_train_data import SceneCache
        cache = SceneCache("cache", enable_shuffle=configs.enable_shuffle)
        exposures, summaries = load_all_exposures(cache, scene_name, configs.num_shots, H_orig, W_orig)
        
        # Start with reference exposure
        ref_idx = configs.num_shots // 2
        current_output = exposures[ref_idx].unsqueeze(0).to(device)
        current_sum = summaries[ref_idx].unsqueeze(0).to(device)
        
        scene_results = []
        
        # Define merging order (skip reference index)
        merge_order = [i for i in range(configs.num_shots) if i != ref_idx]
        
        for step, exp_idx in enumerate(merge_order):
            exp_img = exposures[exp_idx].unsqueeze(0).to(device)
            exp_sum = summaries[exp_idx].unsqueeze(0).to(device)
            
            # Merge current output with next exposure
            result = process_image_tiled(
                model, current_output, exp_img, current_sum, exp_sum,
                patch_h, patch_w, device
            )
            
            # Unshuffle if needed
            if configs.enable_shuffle and unshuffle_indices.numel() > 0:
                unshuffle_np = unshuffle_indices[0].cpu().numpy()
                result_unshuffled = unshuffle_output(result, unshuffle_np, H_orig, W_orig)
                ref_unshuffled = unshuffle_output(ref_HDR, unshuffle_np, H_orig, W_orig)
            else:
                result_unshuffled = result
                ref_unshuffled = ref_HDR
            
            # Calculate loss against ground truth
            loss, loss_dict = criterion(result_unshuffled, ref_unshuffled)
            scene_results.append({
                'step': step,
                'exposure_idx': exp_idx,
                'loss': loss.item(),
                'loss_dict': {k: v for k, v in loss_dict.items()}
            })
            
            diagnostic_name = f"Iterative - Scene {scene_name}, Step {step} (Exp {exp_idx})"
            comprehensive_diagnostics(
                result_unshuffled, ref_unshuffled,
                exp_img, current_output, exp_sum, current_sum,
                name=diagnostic_name
            )
            
            # Save intermediate outputs - dump_sample only takes 2 arguments
            step_output_dir = scene_dir / f"step_{step:02d}_exp_{exp_idx:02d}"
            step_output_dir.mkdir(parents=True, exist_ok=True)
            dump_sample(str(step_output_dir), result_unshuffled.cpu().numpy())
            
            
            save_diagnostic_images(
                result_unshuffled, ref_unshuffled,
                exp_img, current_output,
                str(step_output_dir),
                idx=step
            )
            
            # Save comparison image (in parent scene directory to avoid conflicts)
            comparison_path = scene_dir / f"comparison_step_{step:02d}.png"
            save_comparison_image(
                result_unshuffled, ref_unshuffled,
                str(comparison_path),
                label1=f'Step {step} (Exp {exp_idx})',
                label2='Ground Truth'
            )
            
            # Format loss components for display
            loss_components = ', '.join([f'{k}={v:.4f}' for k, v in loss_dict.items()])
            print(f"  Scene {scene_name} | Step {step} (Exp {exp_idx}) → Loss: {loss.item():.6f} ({loss_components})")
            
            # Update current output for next iteration
            # Re-shuffle the output if shuffling is enabled
            if configs.enable_shuffle and unshuffle_indices.numel() > 0:
                # We need to shuffle the result back before using it as input
                from utils.hdr_load_train_data import shuffle_image
                shuffle_indices, _ = cache.load_shuffle_pattern(scene_name)
                
                # Convert to numpy, shuffle, convert back
                result_np = result_unshuffled[0].cpu().numpy().transpose(1, 2, 0)
                result_shuffled_np = shuffle_image(result_np, shuffle_indices, H_orig, W_orig)
                current_output = torch.from_numpy(result_shuffled_np).permute(2, 0, 1).unsqueeze(0).to(device)
            else:
                current_output = result
            
            # Generate summary for current output (simple resize)
            current_output_np = current_output[0].cpu().numpy().transpose(1, 2, 0)
            current_sum_np = cv2.resize(current_output_np, configs.patch_size[::-1], 
                                       interpolation=cv2.INTER_AREA)
            current_sum = torch.from_numpy(current_sum_np).permute(2, 0, 1).unsqueeze(0).to(device)
        
        # Save final result - dump_sample only takes 2 arguments
        final_output_dir = scene_dir / "final_result"
        final_output_dir.mkdir(parents=True, exist_ok=True)
        dump_sample(str(final_output_dir), result_unshuffled.cpu().numpy())
        
        # Save scene summary
        summary_path = scene_dir / "summary.txt"
        with open(summary_path, 'w') as f:
            f.write(f"Scene: {scene_name}\n")
            f.write(f"Mode: Iterative Merging\n")
            f.write(f"Starting with: Exposure {ref_idx} (reference)\n")
            f.write("="*60 + "\n")
            for res in scene_results:
                f.write(f"Step {res['step']} (Exp {res['exposure_idx']}): Total Loss = {res['loss']:.6f}\n")
                # Write loss components if available
                if 'loss_dict' in res:
                    for key, val in res['loss_dict'].items():
                        f.write(f"  {key}: {val:.6f}\n")
            final_loss = scene_results[-1]['loss']
            f.write("="*60 + "\n")
            f.write(f"Final Loss: {final_loss:.6f}\n")
        
        results.append({
            'scene': scene_name,
            'results': scene_results
        })
    
    return results


def print_summary(results, mode_name):
    """Print and save summary statistics"""
    print(f"\n{'='*80}")
    print(f"SUMMARY: {mode_name}")
    print(f"{'='*80}")
    
    all_losses = []
    for scene_result in results:
        scene_losses = [r['loss'] for r in scene_result['results']]
        all_losses.extend(scene_losses)
        
        print(f"\nScene: {scene_result['scene']}")
        print(f"  Mean Loss: {np.mean(scene_losses):.6f}")
        print(f"  Std Loss:  {np.std(scene_losses):.6f}")
        print(f"  Min Loss:  {np.min(scene_losses):.6f}")
        print(f"  Max Loss:  {np.max(scene_losses):.6f}")
    
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Mean Loss: {np.mean(all_losses):.6f}")
    print(f"Std Loss:  {np.std(all_losses):.6f}")
    print(f"Min Loss:  {np.min(all_losses):.6f}")
    print(f"Max Loss:  {np.max(all_losses):.6f}")
    print(f"{'='*80}\n")
    
    return {
        'mean': np.mean(all_losses),
        'std': np.std(all_losses),
        'min': np.min(all_losses),
        'max': np.max(all_losses)
    }


# =============================================================================
# Main Testing Script
# =============================================================================

# Get configurations
configs = Configs()

# Add default configs if missing
if not hasattr(configs, 'enable_shuffle'):
    configs.enable_shuffle = False

print("="*80)
print("U2Net HDR Testing")
print("="*80)
print(f"Pixel Shuffling: {'ENABLED' if configs.enable_shuffle else 'DISABLED'}")

# Load test dataset
print("\nLoading test dataset...")
test_dataset = U2NetTestDataset(configs=configs, enable_shuffle=configs.enable_shuffle)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False,
    num_workers=configs.num_workers if hasattr(configs, 'num_workers') else 4,
    prefetch_factor=4
)
print(f"Test dataset loaded: {len(test_dataset)} scenes")

# Build model
print("\nBuilding U2Net model...")
dim = configs.dim
img1_dim = configs.c_dim
img2_dim = configs.c_dim
H = configs.patch_size[0]
W = configs.patch_size[1]

model = U2Net(dim=dim, img1_dim=img1_dim, img2_dim=img2_dim, H=H, W=W)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

# Setup device
if configs.multigpu is False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)

print(f"Using device: {device}")

# Define criterion
criterion = HDRLoss(use_shuffling=configs.enable_shuffle, 
                   MU=configs.MU if hasattr(configs, 'MU') else 5000.0)

# Load checkpoint
checkpoint_file = os.path.join(configs.checkpoint_dir, 'checkpoint.tar')
best_checkpoint_file = os.path.join(configs.checkpoint_dir, 'best_checkpoint.tar')

if os.path.isfile(best_checkpoint_file):
    print(f"\nLoading best checkpoint from {best_checkpoint_file}")
    checkpoint = torch.load(best_checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f})")
elif os.path.isfile(checkpoint_file):
    print(f"\nLoading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f})")
else:
    raise FileNotFoundError(f'No checkpoint files found in {configs.checkpoint_dir}')

# Enable DataParallel if needed
if configs.multigpu is True:
    model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

# Create output directory
output_dir = Path(configs.sample_dir) / "comprehensive_test"
output_dir.mkdir(parents=True, exist_ok=True)


def main():
    """Main testing function"""
    print("\n" + "="*80)
    print("Starting Comprehensive Testing")
    print("="*80)
    
    # Test Mode A: Individual merging
    results_individual = test_mode_individual(
        model, test_dataloader, configs, device, str(output_dir)
    )
    stats_individual = print_summary(results_individual, "Individual Merging")
    
    # Test Mode B: Iterative merging
    results_iterative = test_mode_iterative(
        model, test_dataloader, configs, device, str(output_dir)
    )
    stats_iterative = print_summary(results_iterative, "Iterative Merging")
    
    # Save overall summary
    summary_file = output_dir / "overall_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("U2Net Comprehensive Testing Results\n")
        f.write("="*80 + "\n\n")
        
        f.write("MODE A: INDIVIDUAL MERGING\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Loss: {stats_individual['mean']:.6f}\n")
        f.write(f"Std Loss:  {stats_individual['std']:.6f}\n")
        f.write(f"Min Loss:  {stats_individual['min']:.6f}\n")
        f.write(f"Max Loss:  {stats_individual['max']:.6f}\n\n")
        
        f.write("MODE B: ITERATIVE MERGING\n")
        f.write("-"*80 + "\n")
        f.write(f"Mean Loss: {stats_iterative['mean']:.6f}\n")
        f.write(f"Std Loss:  {stats_iterative['std']:.6f}\n")
        f.write(f"Min Loss:  {stats_iterative['min']:.6f}\n")
        f.write(f"Max Loss:  {stats_iterative['max']:.6f}\n")
    
    print(f"\nOverall summary saved to {summary_file}")
    print("\nTesting completed successfully!")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise