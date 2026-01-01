import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.tools import *
from model.u2net import U2Net
import numpy as np
from config import Configs
import random
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from dataset_diag import *
import gc
from torch.cuda.amp import autocast, GradScaler
import math

# Import the modified dataset with shuffling support
from utils.hdr_load_train_data import (
    U2NetDataset, 
    U2NetTestDataset, 
    unshuffle_output
)


def setup_seed(seed=0):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Logger:
    """Simple logger for training"""
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.log_file = os.path.join(log_dir, 'training.log')
        
    def log(self, message, print_console=True):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        
        if print_console:
            print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')


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


def eval_one_epoch(epoch, save_outputs=True, enable_shuffle=True):
    """Evaluation with reflection padding for clean borders"""
    model.eval()
    count = 0
    total_loss = 0
    total_loss_dict = {}

    patch_h, patch_w = configs.patch_size
    stride = patch_h // 2
    PATCH_CHUNK = 100
    pad_size = stride // 2

    pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{configs.epoch} [Eval]')

    for batch_idx, data in enumerate(pbar):
        sample_path, img1, img2, sum1, sum2, ref_HDR, unshuffle_indices, (H, W) = data
        sample_path = sample_path[0]

        img1 = img1.to(device)
        img2 = img2.to(device)
        sum1 = sum1.to(device)
        sum2 = sum2.to(device)
        ref_HDR = ref_HDR.to(device)

        _, _, H_tensor, W_tensor = img1.shape

        if H_tensor > patch_h or W_tensor > patch_w:
            img1_pad = F.pad(img1, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            img2_pad = F.pad(img2, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            
            _, _, H_pad, W_pad = img1_pad.shape
            
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

            result_pad = torch.zeros((1, ref_HDR.shape[1], H_pad, W_pad), 
                                    device=device, dtype=torch.float32)
            weight_map = torch.zeros((1, 1, H_pad, W_pad), device=device)

            blend = make_blend_window(patch_h, patch_w, device, blend_type='hann')

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

                    out = model(b_img1, b_img2, b_sum1, b_sum2)
                    out = out * blend

                    for i, (hs, ws) in enumerate(batch):
                        he = hs + patch_h
                        we = ws + patch_w
                        result_pad[:, :, hs:he, ws:we] += out[i:i+1]
                        weight_map[:, :, hs:he, ws:we] += blend

            result_pad = result_pad / torch.clamp(weight_map, min=1e-3)
            result = result_pad[:, :, pad_size:-pad_size, pad_size:-pad_size]
        else:
            img1_pad = F.pad(img1, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            img2_pad = F.pad(img2, (pad_size, pad_size, pad_size, pad_size), mode='reflect')
            
            with torch.no_grad():
                result_pad = model(img1_pad, img2_pad, sum1, sum2)
                result = result_pad[:, :, pad_size:-pad_size, pad_size:-pad_size]

        if enable_shuffle and unshuffle_indices.numel() > 0:
            unshuffle_np = unshuffle_indices[0].cpu().numpy()
            H_orig = H[0].item()
            W_orig = W[0].item()
            
            result_unshuffled = unshuffle_output(result, unshuffle_np, H_orig, W_orig)
            ref_HDR_unshuffled = unshuffle_output(ref_HDR, unshuffle_np, H_orig, W_orig)
        else:
            result_unshuffled = result
            ref_HDR_unshuffled = ref_HDR

        loss, loss_dict = criterion(result, ref_HDR)
        total_loss += loss.item()
        
        for key, value in loss_dict.items():
            total_loss_dict[key] = total_loss_dict.get(key, 0.0) + value
        
        count += 1
        
        if save_outputs:
            comprehensive_diagnostics(
                result_unshuffled, ref_HDR_unshuffled, 
                img1, img2, sum1, sum2,
                name=f"Eval - Epoch {epoch+1}, Scene {batch_idx}"
            )
            save_diagnostic_images(result_unshuffled, ref_HDR_unshuffled, 
                                  img1, img2, sample_path, batch_idx)
            from utils.tools import dump_sample
            dump_sample(sample_path, result_unshuffled.cpu().numpy())
            
            result_np = result_unshuffled.detach().cpu().numpy()[0]
            ref_HDR_np = ref_HDR_unshuffled.detach().cpu().numpy()[0]
            
            output_01 = np.clip((result_np + 1) / 2, 0, 1)
            ref_01 = np.clip((ref_HDR_np + 1) / 2, 0, 1)
            
            output_01 = np.transpose(output_01, (1, 2, 0))
            ref_01 = np.transpose(ref_01, (1, 2, 0))
            
            output_8bit = (output_01 * 255).astype(np.uint8)
            ref_8bit = (ref_01 * 255).astype(np.uint8)
            
            h, w = output_8bit.shape[:2]
            comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
            comparison[:, :w] = output_8bit
            comparison[:, w:] = ref_8bit
            
            cv2.putText(comparison, 'Output', (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, 'Reference', (w+10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imwrite(os.path.join(sample_path, 'side_by_side.png'), 
                        comparison[..., ::-1])
    
    avg_loss_dict = {key: value / count for key, value in total_loss_dict.items()}
    return total_loss / max(count, 1), avg_loss_dict


def print_grad_stats(model, prefix=""):
    """Print gradient statistics"""
    total_norm_sq = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None:
            gnorm = p.grad.data.norm(2).item()
            total_norm_sq += gnorm ** 2
    total_norm = total_norm_sq ** 0.5
    print(f"{prefix} TOTAL_GRAD_NORM = {total_norm:.6f}")
    return total_norm
    
    
def train_one_epoch(epoch):
    """Enhanced version with comprehensive logging"""
    model.train()
    epoch_loss = 0.0
    epoch_loss_dict = {}
    skipped_batches = 0
    grad_norms = []  # ✅ Track gradient norms

    pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{configs.epoch} [Train]')

    for idx, data in enumerate(pbar):
        img1, img2, sum1, sum2, ref_HDR = data

        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        sum1 = sum1.to(device, non_blocking=True)
        sum2 = sum2.to(device, non_blocking=True)
        ref_HDR = ref_HDR.to(device, non_blocking=True)

        optimizer.zero_grad()

        # Forward pass
        if configs.use_mixed_precision:
            with autocast():
                result = model(img1, img2, sum1, sum2)
                loss, loss_dict = criterion(result, ref_HDR)
        else:
            result = model(img1, img2, sum1, sum2)
            loss, loss_dict = criterion(result, ref_HDR)

        # Check validity
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() > 1e5:
            logger.log(f"\n⚠️  Skipping batch {idx} - Invalid loss: {loss.item():.4f}")
            skipped_batches += 1
            optimizer.zero_grad()
            continue
        
        if not loss.requires_grad:
            logger.log(f"\n⚠️  Skipping batch {idx} - No gradient connection")
            skipped_batches += 1
            optimizer.zero_grad()
            continue

        # Backward pass
        if configs.use_mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # ✅ Track gradient norm BEFORE clipping
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            
            if idx % 10 == 0:
                print(f"[Epoch {epoch+1} | Batch {idx}] GRAD_NORM = {total_norm:.6f}")
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                          max_norm=configs.gradient_clip_norm)
            
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            
            # Track gradient norm
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            grad_norms.append(total_norm)
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                          max_norm=configs.gradient_clip_norm)
            optimizer.step()

        # Update metrics
        epoch_loss += loss.item()
        for key, value in loss_dict.items():
            epoch_loss_dict[key] = epoch_loss_dict.get(key, 0.0) + value

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'l1': f'{loss_dict["l1"]:.4f}',
            'pred': f'[{result.min():.2f},{result.max():.2f}]',
            'grad': f'{total_norm:.2f}'  # ✅ Show gradient norm
        })

        # TensorBoard logging
        if idx % 10 == 0:
            global_step = epoch * len(train_dataloader) + idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
            writer.add_scalar('Train/L1Loss', loss_dict['l1'], global_step)
            writer.add_scalar('Train/GradNorm', total_norm, global_step)
            writer.add_scalar('Train/PredRange', 
                            result.max().item() - result.min().item(), global_step)

    # ✅ Epoch summary statistics
    if skipped_batches > 0:
        logger.log(f"⚠️  Skipped {skipped_batches}/{len(train_dataloader)} batches")

    num_valid_batches = len(train_dataloader) - skipped_batches
    avg_loss = epoch_loss / max(num_valid_batches, 1)
    avg_loss_dict = {
        key: value / max(num_valid_batches, 1)
        for key, value in epoch_loss_dict.items()
    }
    
    # ✅ Gradient statistics
    if grad_norms:
        avg_grad = np.mean(grad_norms)
        max_grad = np.max(grad_norms)
        logger.log(f"Gradient Stats - Avg: {avg_grad:.4f}, Max: {max_grad:.4f}, "
                   f"Clip: {configs.gradient_clip_norm}")
        
        # Warning if clipping too much
        if avg_grad > configs.gradient_clip_norm * 0.8:
            logger.log(f"⚠️  Average gradient near clip threshold - "
                      f"consider increasing clip norm")

    return avg_loss, avg_loss_dict


# =============================================================================
# Main Training Script
# =============================================================================

# Get configurations
configs = Configs()

# Add default configs if missing
if not hasattr(configs, 'enable_shuffle'):
    configs.enable_shuffle = False
if not hasattr(configs, 'use_mixed_precision'):
    configs.use_mixed_precision = True  # Enable by default
if not hasattr(configs, 'gradient_clip_norm'):
    configs.gradient_clip_norm = 1.0

# Create directories
os.makedirs(configs.checkpoint_dir, exist_ok=True)
os.makedirs(configs.sample_dir, exist_ok=True)
log_dir = os.path.join(configs.checkpoint_dir, 'logs')
os.makedirs(log_dir, exist_ok=True)

# Initialize logger and tensorboard
logger = Logger(log_dir)
writer = SummaryWriter(log_dir=log_dir)

logger.log("="*80)
logger.log("Starting U2Net HDR Training")
logger.log("="*80)

# Setup reproducibility
if hasattr(configs, 'seed'):
    setup_seed(configs.seed)
    logger.log(f"Random seed set to: {configs.seed}")

# Log training configuration
logger.log(f"Pixel Shuffling: {'ENABLED' if configs.enable_shuffle else 'DISABLED'}")
logger.log(f"Mixed Precision: {'ENABLED' if configs.use_mixed_precision else 'DISABLED'}")
logger.log(f"Gradient Clipping: {configs.gradient_clip_norm}")

# Load datasets
logger.log("Loading training dataset...")
train_dataset = U2NetDataset(configs=configs, enable_shuffle=configs.enable_shuffle)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=configs.batch_size, 
    shuffle=True,
    num_workers=configs.num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)
logger.log(f"Training dataset loaded: {len(train_dataset)} patches")

logger.log("Loading test dataset...")
test_dataset = U2NetTestDataset(configs=configs, enable_shuffle=configs.enable_shuffle)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False,
    num_workers=configs.num_workers,
    prefetch_factor=4
)
logger.log(f"Test dataset loaded: {len(test_dataset)} scenes")

# Build model
logger.log("Building U2Net model...")
dim = configs.dim
img1_dim = configs.c_dim
img2_dim = configs.c_dim
H = configs.patch_size[0]
W = configs.patch_size[1]

model = U2Net(dim=dim, img1_dim=img1_dim, img2_dim=img2_dim, H=H, W=W)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logger.log(f"Total parameters: {total_params:,}")
logger.log(f"Trainable parameters: {trainable_params:,}")

# Setup device
if configs.multigpu is False:
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == torch.device('cpu'):
        raise EnvironmentError('No GPUs, cannot initialize multigpu training.')
    model.to(device)

logger.log(f"Using device: {device}")

# ✅ CRITICAL: Create GradScaler ONCE globally, not in train_one_epoch
scaler = GradScaler(
    init_scale=2.**10,      # Start moderate
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
    enabled=configs.use_mixed_precision
)

# Define optimizer
optimizer = optim.AdamW(
    model.parameters(),
    betas=(configs.beta1, configs.beta2),
    lr=configs.learning_rate,
    weight_decay=1e-4
)

logger.log(f"Optimizer: AdamW (lr={configs.learning_rate})")

# Define criterion
criterion = HDRLoss(use_shuffling=configs.enable_shuffle, MU=configs.MU if hasattr(configs, 'MU') else 5000.0)

# Learning rate scheduler
steps_per_epoch = len(train_dataloader)
total_steps = configs.epoch * steps_per_epoch
warmup_steps = configs.lr_warmup_epochs * steps_per_epoch
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=5,           # Restart every 50 epochs
    T_mult=1,         # Keep same cycle length
    eta_min=1e-5      # Minimum LR
)

# Load checkpoint
start_epoch = 0
best_loss = float('inf')
checkpoint_file = os.path.join(configs.checkpoint_dir, 'checkpoint.tar')

if os.path.isfile(checkpoint_file):
    logger.log(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    lr_scheduler.load_state_dict(checkpoint['scheduler'])
    if 'best_loss' in checkpoint:
        best_loss = checkpoint['best_loss']
    # ✅ Load scaler state if available
    if 'scaler' in checkpoint and configs.use_mixed_precision:
        scaler.load_state_dict(checkpoint['scaler'])
        
    logger.log(f"Loaded checkpoint (epoch {start_epoch}, best loss: {best_loss:.6f})")
else:
    logger.log("No checkpoint found, starting from scratch")

# Enable DataParallel if needed
if configs.multigpu is True:
    model = torch.nn.DataParallel(model)
    logger.log(f"Using {torch.cuda.device_count()} GPUs")


def save_checkpoint(epoch, loss, best_loss, optimizer, model, lr_scheduler, is_best=False):
    """Save checkpoint with scaler state"""
    if configs.multigpu is False:
        model_state = model.state_dict()
    else:
        model_state = model.module.state_dict()
    
    save_dict = {
        'epoch': epoch + 1,
        'loss': loss,
        'best_loss': best_loss,
        'optimizer_state_dict': optimizer.state_dict(),
        'model_state_dict': model_state,
        'scheduler': lr_scheduler.state_dict(),
        'scaler': scaler.state_dict(),  # ✅ Save scaler state
        'enable_shuffle': configs.enable_shuffle,
    }
    
    checkpoint_path = os.path.join(configs.checkpoint_dir, 'checkpoint.tar')
    torch.save(save_dict, checkpoint_path)
    
    if (epoch + 1) % 5 == 0:
        epoch_checkpoint_path = os.path.join(configs.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.tar')
        torch.save(save_dict, epoch_checkpoint_path)
    
    if is_best:
        best_checkpoint_path = os.path.join(configs.checkpoint_dir, 'best_checkpoint.tar')
        torch.save(save_dict, best_checkpoint_path)
    
    return checkpoint_path



def train(start_epoch):
    """Main training loop with health checks"""
    global best_loss
    
    logger.log("\n" + "="*80)
    logger.log("Training Configuration:")
    logger.log(f"  Shuffle enabled: {configs.enable_shuffle}")
    logger.log(f"  Mixed precision: {configs.use_mixed_precision}")
    logger.log(f"  Gradient clip norm: {configs.gradient_clip_norm}")
    logger.log(f"  Max LR: {configs.max_lr}")
    logger.log(f"  Epochs: {configs.epoch}")
    logger.log(f"  Batch size: {configs.batch_size}")
    logger.log("="*80 + "\n")
    
    training_start_time = time.time()
    recent_losses = []  # Track last 5 losses
    
    for epoch in range(start_epoch, configs.epoch):
        epoch_start_time = time.time()
        
        logger.log(f"\n{'='*80}")
        logger.log(f"Epoch [{epoch + 1}/{configs.epoch}]")
        logger.log(f"{'='*80}")
        
        # ✅ Log current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        logger.log(f"Learning Rate: {current_lr:.8f}")
        
        # Train
        train_loss, train_loss_dict = train_one_epoch(epoch)
        logger.log(f"Train Loss: {train_loss:.8f}")
        
        # ✅ Log loss components
        logger.log(f"Loss Components:")
        for key, value in train_loss_dict.items():
            logger.log(f"  {key}: {value:.8f}")
        
        # Track recent losses
        recent_losses.append(train_loss)
        if len(recent_losses) > 5:
            recent_losses.pop(0)
        
        # ✅ Health check every 5 epochs
        if (epoch + 1) % 5 == 0 and len(recent_losses) == 5:
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            logger.log(f"\nHealth Check:")
            logger.log(f"  Recent 5 epoch loss: {loss_mean:.6f} ± {loss_std:.6f}")
            
            if loss_std < 5e-4:
                logger.log(f"  ⚠️  Loss plateau detected (std={loss_std:.6f})")
                logger.log(f"  Consider: Increase gradient_clip_norm or adjust LR")
        
        # TensorBoard
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('Train/LearningRate', current_lr, epoch)
        
        # Step scheduler
        lr_scheduler.step()
        
        # Save checkpoint
        save_checkpoint(epoch, train_loss, best_loss, optimizer, model, lr_scheduler, False)
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Evaluate
        if (epoch + 1) % 50 == 0:
            with torch.no_grad():
                eval_loss, eval_loss_dict = eval_one_epoch(
                    epoch, 
                    enable_shuffle=configs.enable_shuffle
                )
                logger.log(f"Eval Loss: {eval_loss:.8f}")
                writer.add_scalar('Eval/Loss', eval_loss, epoch)
        
                is_best = eval_loss < best_loss
                if is_best:
                    logger.log(f"🎉 New best! {best_loss:.8f} → {eval_loss:.8f}")
                    best_loss = eval_loss
                    save_checkpoint(epoch, eval_loss, best_loss, optimizer, model, lr_scheduler, True)
        
        epoch_time = time.time() - epoch_start_time
        logger.log(f"Epoch time: {epoch_time:.2f}s")
    
    total_time = time.time() - training_start_time
    logger.log("\n" + "="*80)
    logger.log("Training Completed!")
    logger.log(f"Total time: {total_time/3600:.2f} hours")
    logger.log(f"Best loss: {best_loss:.8f}")
    logger.log("="*80)
    
    writer.close()


if __name__ == '__main__':
    try:
        logger.log("Starting training...")
        train(start_epoch)
        logger.log("Training finished successfully!")
    except Exception as e:
        logger.log(f"Error: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        raise
    finally:
        writer.close()