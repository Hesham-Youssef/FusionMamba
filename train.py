import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.tools import *
from model.u2net import U2Net
import numpy as np
from utils.hdr_load_train_data import *
from config import Configs
import random
import time
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import json


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
    """
    Create blending window optimized to eliminate tiling artifacts
    
    Args:
        blend_type: 'gaussian', 'cosine', 'linear', or 'hann'
    """
    if blend_type == 'gaussian':
        # Gaussian - BEST for eliminating grid artifacts
        y = torch.linspace(-1, 1, h, device=device)
        x = torch.linspace(-1, 1, w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing='ij')
        
        # Gaussian from center with sigma tuned to minimize artifacts
        dist = torch.sqrt(xx**2 + yy**2)
        sigma = 0.5  # Larger = more gradual falloff
        weight = torch.exp(-(dist**2) / (2 * sigma**2))
        
        # Normalize to [0, 1] range
        weight = (weight - weight.min()) / (weight.max() - weight.min() + 1e-8)
        return weight[None, None]
        
    elif blend_type == 'cosine':
        # Cosine window - smoother than Hann
        wy = (1 - torch.cos(torch.linspace(0, torch.pi, h, device=device))) / 2
        wx = (1 - torch.cos(torch.linspace(0, torch.pi, w, device=device))) / 2
    elif blend_type == 'linear':
        # Linear ramps
        wy = torch.linspace(0, 1, h, device=device)
        wx = torch.linspace(0, 1, w, device=device)
    else:  # hann
        wy = torch.hann_window(h, periodic=False, device=device)
        wx = torch.hann_window(w, periodic=False, device=device)
    
    w2d = wy[:, None] * wx[None, :]
    return w2d[None, None]  # (1,1,H,W)


def eval_one_epoch(epoch, save_outputs=True):
    model.eval()
    mean_loss = 0.0
    count = 0

    patch_h, patch_w = configs.patch_size
    # FIX: MAXIMUM overlap to eliminate grid artifacts (87.5% overlap)
    stride = patch_h // 2  # Was patch_h // 4
    
    # FIX: Larger padding for better edge context
    PAD = patch_h // 3  # Was patch_h // 4

    PATCH_CHUNK = 120  # tune: 2–8 for 6GB GPU

    pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{configs.epoch} [Eval]')

    for _, data in enumerate(pbar):
        sample_path, img1_ldr, img2_ldr, img1_hdr, img2_hdr, sum1, sum2, ref_HDR = data
        sample_path = sample_path[0]

        img1_ldr = img1_ldr.to(device)
        img2_ldr = img2_ldr.to(device)
        img1_hdr = img1_hdr.to(device)
        img2_hdr = img2_hdr.to(device)
        sum1 = sum1.to(device)
        sum2 = sum2.to(device)
        ref_HDR = ref_HDR.to(device)

        _, _, H, W = img1_ldr.shape

        # ============================================================
        # TILED PATH WITH PROPER PADDING
        # ============================================================
        if H > patch_h or W > patch_w:

            img1 = torch.cat([img1_ldr, img1_hdr], dim=1)
            img2 = torch.cat([img2_ldr, img2_hdr], dim=1)

            # FIX: Pad images with reflection for better edge handling
            img1_padded = F.pad(img1, (PAD, PAD, PAD, PAD), mode='reflect')
            img2_padded = F.pad(img2, (PAD, PAD, PAD, PAD), mode='reflect')
            sum1_padded = F.pad(sum1, (PAD, PAD, PAD, PAD), mode='reflect')
            sum2_padded = F.pad(sum2, (PAD, PAD, PAD, PAD), mode='reflect')

            H_padded = H + 2 * PAD
            W_padded = W + 2 * PAD

            # ---- Generate patch coordinates with better coverage ----
            h_starts = list(range(0, max(1, H_padded - patch_h + 1), stride))
            w_starts = list(range(0, max(1, W_padded - patch_w + 1), stride))

            # FIX: Ensure complete coverage by adding final patches if needed
            if h_starts[-1] + patch_h < H_padded:
                h_starts.append(H_padded - patch_h)
            if w_starts[-1] + patch_w < W_padded:
                w_starts.append(W_padded - patch_w)

            patch_coords = [(hs, ws) for hs in h_starts for ws in w_starts]

            # Initialize output buffers (padded size)
            result = torch.zeros((1, ref_HDR.shape[1], H_padded, W_padded), 
                                device=device, dtype=ref_HDR.dtype)
            weight_map = torch.zeros((1, 1, H_padded, W_padded), device=device)

            # FIX: Use Gaussian window for smoothest blending (eliminates grid artifacts)
            blend = make_blend_window(patch_h, patch_w, device, blend_type='gaussian')

            # ---- Chunked inference ----
            with torch.no_grad():
                for start in range(0, len(patch_coords), PATCH_CHUNK):
                    batch = patch_coords[start:start + PATCH_CHUNK]

                    b_img1, b_img2, b_sum1, b_sum2 = [], [], [], []

                    for hs, ws in batch:
                        he = hs + patch_h
                        we = ws + patch_w

                        b_img1.append(img1_padded[:, :, hs:he, ws:we])
                        b_img2.append(img2_padded[:, :, hs:he, ws:we])
                        b_sum1.append(sum1_padded[:, :, hs:he, ws:we])
                        b_sum2.append(sum2_padded[:, :, hs:he, ws:we])

                    b_img1 = torch.cat(b_img1, dim=0)
                    b_img2 = torch.cat(b_img2, dim=0)
                    b_sum1 = torch.cat(b_sum1, dim=0)
                    b_sum2 = torch.cat(b_sum2, dim=0)

                    out = model(b_img1, b_img2, b_sum1, b_sum2)
                    
                    # Apply blending window
                    out = out * blend

                    # Accumulate results
                    for i, (hs, ws) in enumerate(batch):
                        he = hs + patch_h
                        we = ws + patch_w

                        result[:, :, hs:he, ws:we] += out[i:i+1]
                        weight_map[:, :, hs:he, ws:we] += blend

            # FIX: Normalize with minimum threshold
            result = result / torch.clamp(weight_map, min=1e-3)
            
            # FIX: Remove padding to get back to original size
            result = result[:, :, PAD:PAD+H, PAD:PAD+W]
            
            # FIX: Optional post-smoothing to eliminate remaining artifacts
            # Uncomment if you still see grid patterns
            # result = self.smooth_boundaries(result, patch_h, stride)

        # ============================================================
        # NON-TILED PATH
        # ============================================================
        else:
            img1 = torch.cat([img1_ldr, img1_hdr], dim=1)
            img2 = torch.cat([img2_ldr, img2_hdr], dim=1)

            with torch.no_grad():
                result = model(img1, img2, sum1, sum2)

        # ============================================================
        # LOSS + OUTPUT
        # ============================================================
        loss = criterion(tonemap(result), tonemap(ref_HDR))

        if save_outputs:
            dump_sample(sample_path, result.cpu().numpy())

        mean_loss += loss.item()
        count += 1
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    return mean_loss / max(count, 1)


# Get configurations
configs = Configs()

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

# Load Data & build dataset
logger.log("Loading training dataset...")
train_dataset = U2NetDataset(configs=configs)
train_dataloader = DataLoader(
    train_dataset, 
    batch_size=configs.batch_size, 
    shuffle=True,
    num_workers=configs.num_workers if hasattr(configs, 'num_workers') else 4,
    pin_memory=True
)
logger.log(f"Training dataset loaded: {len(train_dataset)} patches")

logger.log("Loading test dataset...")
test_dataset = U2NetTestDataset(configs=configs)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False
)
logger.log(f"Test dataset loaded: {len(test_dataset)} scenes")

# Build U2Net model
logger.log("Building U2Net model...")
dim = configs.dim if hasattr(configs, 'dim') else 32
img1_dim = configs.c_dim * 2
img2_dim = configs.c_dim * 2
H = configs.patch_size[0] if hasattr(configs, 'patch_size') else 64
W = configs.patch_size[1] if hasattr(configs, 'patch_size') else 64

model = U2Net(dim=dim, img1_dim=img1_dim, img2_dim=img2_dim, H=H, W=W)

# Count parameters
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

# Define optimizer
optimizer = optim.Adam(
    model.parameters(), 
    betas=(configs.beta1, configs.beta2), 
    lr=configs.learning_rate
)
logger.log(f"Optimizer: Adam (lr={configs.learning_rate}, betas=({configs.beta1}, {configs.beta2}))")

# Define Criterion
criterion = HDRLoss()

# Define Scheduler
lr_scheduler = PolyLR(optimizer, max_iter=configs.epoch, power=0.9)

# Read checkpoints
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
    logger.log(f"Loaded checkpoint (epoch {start_epoch}, best loss: {best_loss:.6f})")

    eval_loss = eval_one_epoch(0)
    logger.log(f"Eval Loss: {eval_loss:.8f}")
else:
    logger.log("No checkpoint found, starting from scratch")

# Enable DataParallel if needed
if configs.multigpu is True:
    model = torch.nn.DataParallel(model)
    logger.log(f"Using {torch.cuda.device_count()} GPUs with DataParallel")


def train_one_epoch(epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0.0
    
    # Progress bar
    pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{configs.epoch} [Train]')
    
    for idx, data in enumerate(pbar):
        # Unpack data
        img1_ldr, img2_ldr, img1_hdr, img2_hdr, sum1, sum2, ref_HDR = data
        
        # Move to device
        img1_ldr = img1_ldr.to(device)
        img2_ldr = img2_ldr.to(device)
        img1_hdr = img1_hdr.to(device)
        img2_hdr = img2_hdr.to(device)
        sum1 = sum1.to(device)
        sum2 = sum2.to(device)
        ref_HDR = ref_HDR.to(device)
        
        # Prepare inputs
        img1 = torch.cat([img1_ldr, img1_hdr], dim=1)
        img2 = torch.cat([img2_ldr, img2_hdr], dim=1)
        
        # Forward pass
        result = model(img1, img2, sum1, sum2)
        
        # Check for NaN/Inf in output (useful for debugging)
        if torch.isnan(result).any() or torch.isinf(result).any():
            logger.log(f"⚠️  Warning: Invalid values in model output at batch {idx+1}")
            logger.log(f"   NaN count: {torch.isnan(result).sum().item()}")
            logger.log(f"   Inf count: {torch.isinf(result).sum().item()}")
            # Skip this batch
            continue
        
        # Compute loss
        loss = criterion(tonemap(result), tonemap(ref_HDR))
        
        # Check if loss is valid
        if torch.isnan(loss) or torch.isinf(loss):
            logger.log(f"⚠️  Warning: Invalid loss at batch {idx+1}, skipping")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # Log to tensorboard every 100 batches
        if (idx + 1) % 100 == 0:
            global_step = epoch * len(train_dataloader) + idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    avg_loss = epoch_loss / len(train_dataloader)
    return avg_loss


def save_checkpoint(epoch, loss, best_loss, optimizer, model, lr_scheduler, is_best=False):
    """Save checkpoint with proper handling of DataParallel"""
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
        'config': {
            'dim': dim,
            'img1_dim': img1_dim,
            'img2_dim': img2_dim,
            'H': H,
            'W': W,
            'learning_rate': configs.learning_rate,
            'batch_size': configs.batch_size
        }
    }
    
    # Save latest checkpoint
    checkpoint_path = os.path.join(configs.checkpoint_dir, 'checkpoint.tar')
    torch.save(save_dict, checkpoint_path)
    
    # Save epoch checkpoint every N epochs
    if (epoch + 1) % 5 == 0:
        epoch_checkpoint_path = os.path.join(configs.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.tar')
        torch.save(save_dict, epoch_checkpoint_path)
    
    # Save best model
    if is_best:
        best_checkpoint_path = os.path.join(configs.checkpoint_dir, 'best_checkpoint.tar')
        torch.save(save_dict, best_checkpoint_path)
    
    return checkpoint_path


def train(start_epoch):
    """Main training loop"""
    global best_loss
    
    logger.log("\n" + "="*80)
    logger.log("Training Configuration:")
    logger.log(f"  Epochs: {configs.epoch}")
    logger.log(f"  Batch size: {configs.batch_size}")
    logger.log(f"  Learning rate: {configs.learning_rate}")
    logger.log(f"  Patch size: {configs.patch_size}")
    logger.log(f"  Model dimension: {dim}")
    logger.log(f"  Device: {device}")
    logger.log("="*80 + "\n")
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, configs.epoch):
        epoch_start_time = time.time()
        
        # Log epoch header
        logger.log(f"\n{'='*80}")
        logger.log(f"Epoch [{epoch + 1}/{configs.epoch}]")
        logger.log(f"Learning rate: {lr_scheduler.get_last_lr()[0]:.8f}")
        logger.log(f"{'='*80}")
        
        # Train
        train_loss = train_one_epoch(epoch)
        logger.log(f"Train Loss: {train_loss:.8f}")
        
        # Tensorboard logging
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('Train/LearningRate', lr_scheduler.get_last_lr()[0], epoch)
        
        # Update learning rate
        lr_scheduler.step()
        
        # Save checkpoint
        checkpoint_path = save_checkpoint(
            epoch, 0, 0, optimizer, model, lr_scheduler, True
        )
        
        # Evaluate
        eval_loss = eval_one_epoch(epoch)
        logger.log(f"Eval Loss: {eval_loss:.8f}")
        
        # Tensorboard logging
        writer.add_scalar('Eval/Loss', eval_loss, epoch)
        
        # Check if best model
        is_best = eval_loss < best_loss
        if is_best:
            logger.log(f"🎉 New best model! Loss improved from {best_loss:.8f} to {eval_loss:.8f}")
            best_loss = eval_loss
        
        
        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        logger.log(f"Epoch time: {epoch_time:.2f}s")
        logger.log(f"Checkpoint saved to: {checkpoint_path}")
        
        # Save training statistics
        stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'best_loss': best_loss,
            'learning_rate': lr_scheduler.get_last_lr()[0],
            'epoch_time': epoch_time
        }
        
        stats_file = os.path.join(log_dir, 'training_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r') as f:
                all_stats = json.load(f)
        else:
            all_stats = []
        
        all_stats.append(stats)
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=2)
    
    # Training complete
    total_time = time.time() - training_start_time
    logger.log("\n" + "="*80)
    logger.log("Training Completed!")
    logger.log(f"Total training time: {total_time/3600:.2f} hours")
    logger.log(f"Best validation loss: {best_loss:.8f}")
    logger.log("="*80)
    
    writer.close()


if __name__ == '__main__':
    try:
        logger.log("Starting training...")
        train(start_epoch)
        logger.log("Training finished successfully!")
    except Exception as e:
        logger.log(f"Error during training: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
        raise
    finally:
        writer.close()