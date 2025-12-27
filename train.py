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
from dataset_diag import *
import gc


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



def eval_one_epoch(epoch, save_outputs=True, save_tiles=False, save_tile_core=False):
    model.eval()
    mean_loss = 0.0
    count = 0

    patch_h, patch_w = configs.patch_size
    stride = patch_h // 3  # 87.5% overlap
    PATCH_CHUNK = 120

    pbar = tqdm(test_dataloader, desc=f'Epoch {epoch+1}/{configs.epoch} [Eval]')

    for batch_idx, data in enumerate(pbar):
        sample_path, img1, img2, sum1, sum2, ref_HDR = data
        sample_path = sample_path[0]

        img1 = img1.to(device)
        img2 = img2.to(device)
        sum1 = sum1.to(device)
        sum2 = sum2.to(device)
        ref_HDR = ref_HDR.to(device)

        _, _, H, W = img1.shape

        # Tiled inference for large images
        if H > patch_h or W > patch_w:
            # Generate patch positions
            h_starts = []
            h = 0
            while h < H:
                h_start = min(h, H - patch_h)
                if not h_starts or h_starts[-1] != h_start:
                    h_starts.append(h_start)
                if h_start == H - patch_h:
                    break
                h += stride
            
            w_starts = []
            w = 0
            while w < W:
                w_start = min(w, W - patch_w)
                if not w_starts or w_starts[-1] != w_start:
                    w_starts.append(w_start)
                if w_start == W - patch_w:
                    break
                w += stride

            patch_coords = [(hs, ws) for hs in h_starts for ws in w_starts]

            # Result accumulation
            result = torch.zeros((1, ref_HDR.shape[1], H, W), 
                                device=device, dtype=torch.float32)
            weight_map = torch.zeros((1, 1, H, W), device=device)

            # Hann window for blending
            def make_hann_window(h, w):
                hann_h = torch.hann_window(h, device=device).view(-1, 1)
                hann_w = torch.hann_window(w, device=device).view(1, -1)
                window = hann_h * hann_w
                return window.view(1, 1, h, w)
            
            blend = make_hann_window(patch_h, patch_w)

            # Chunked inference
            with torch.no_grad():
                for start in range(0, len(patch_coords), PATCH_CHUNK):
                    batch = patch_coords[start:start + PATCH_CHUNK]
                    batch_size = len(batch)

                    b_img1, b_img2 = [], []
                    for hs, ws in batch:
                        he = hs + patch_h
                        we = ws + patch_w
                        b_img1.append(img1[:, :, hs:he, ws:we])
                        b_img2.append(img2[:, :, hs:he, ws:we])

                    b_img1 = torch.cat(b_img1, dim=0)
                    b_img2 = torch.cat(b_img2, dim=0)
                    
                    # Replicate global summaries
                    b_sum1 = sum1.repeat(batch_size, 1, 1, 1)
                    b_sum2 = sum2.repeat(batch_size, 1, 1, 1)

                    out = model(b_img1, b_img2, b_sum1, b_sum2)
                    out = out * blend

                    for i, (hs, ws) in enumerate(batch):
                        he = hs + patch_h
                        we = ws + patch_w
                        result[:, :, hs:he, ws:we] += out[i:i+1]
                        weight_map[:, :, hs:he, ws:we] += blend

            result = result / torch.clamp(weight_map, min=1e-3)

        else:
            img1 = torch.cat([img1_ldr, img1_hdr], dim=1)
            img2 = torch.cat([img2_ldr, img2_hdr], dim=1)

            with torch.no_grad():
                result = model(img1, img2, sum1, sum2)

        # Compute loss
        loss = criterion(result, ref_HDR)
        
        # Comprehensive diagnostics for first batch
        # if batch_idx == 0:
        comprehensive_diagnostics(
            result, ref_HDR, 
            img1,
            img2,
            sum1, sum2,
            name=f"Eval - Epoch {epoch+1}, Scene {batch_idx}"
        )
        
        # Save outputs and diagnostics
        if save_outputs:
            save_diagnostic_images(
                result, ref_HDR,
                img1, img2,
                sample_path, batch_idx
            )
            
            # Save HDR file
            from utils.tools import dump_sample
            dump_sample(sample_path, result.cpu().numpy())
            
            # Convert to numpy
            result = result.detach().cpu().numpy()[0]  # (C, H, W)
            ref_HDR = ref_HDR.detach().cpu().numpy()[0]
            
            # FIXED: Model output is ALREADY log-compressed (tonemapped)
            # Just map [-1, 1] → [0, 1] for display
            output_01 = (result + 1) / 2
            ref_01 = (ref_HDR + 1) / 2
            
            # Clip and convert to 8-bit
            output_01 = np.clip(output_01, 0, 1)
            ref_01 = np.clip(ref_01, 0, 1)
            
            # Transpose to (H, W, C)
            output_01 = np.transpose(output_01, (1, 2, 0))
            ref_01 = np.transpose(ref_01, (1, 2, 0))
            
            output_8bit = (output_01 * 255).astype(np.uint8)
            ref_8bit = (ref_01 * 255).astype(np.uint8)
            
            # Create side-by-side comparison
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
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)
logger.log(f"Training dataset loaded: {len(train_dataset)} patches")

logger.log("Loading test dataset...")
test_dataset = U2NetTestDataset(configs=configs)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False,
    num_workers=configs.num_workers if hasattr(configs, 'num_workers') else 4,
    prefetch_factor=2
)
logger.log(f"Test dataset loaded: {len(test_dataset)} scenes")

# Build U2Net model
logger.log("Building U2Net model...")
dim = configs.dim if hasattr(configs, 'dim') else 32
img1_dim = configs.c_dim
img2_dim = configs.c_dim
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



def validate_data(data_tuple, name="data"):
    """Check if data contains NaN or Inf"""
    for i, tensor in enumerate(data_tuple):
        if isinstance(tensor, torch.Tensor):
            if torch.isnan(tensor).any():
                print(f"❌ NaN found in {name}[{i}]")
                return False
            if torch.isinf(tensor).any():
                print(f"❌ Inf found in {name}[{i}]")
                return False
            print(f"✓ {name}[{i}]: min={tensor.min():.4f}, max={tensor.max():.4f}, mean={tensor.mean():.4f}")
    return True


from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(epoch):
    model.train()
    epoch_loss = 0.0
    
    scaler = GradScaler()
    accumulation_steps = getattr(configs, 'accumulation_steps', 1)
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{configs.epoch} [Train]')
    
    for idx, data in enumerate(pbar):
        # Unpack data
        img1, img2, sum1, sum2, ref_HDR = data
        
        # Move to device
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        sum1 = sum1.to(device, non_blocking=True)
        sum2 = sum2.to(device, non_blocking=True)
        ref_HDR = ref_HDR.to(device, non_blocking=True)
        
        # Forward pass with mixed precision
        with autocast():
            result = model(img1, img2, sum1, sum2)
            loss = criterion(result, ref_HDR)
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
                

            
        # Backward pass
        scaler.scale(loss).backward()
        
        # Update weights
        if (idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
        
        # Detailed diagnostics every 50 batches
        if idx % 10 == 0:
            with torch.no_grad():
                comprehensive_diagnostics(
                    result.float(), ref_HDR.float(), 
                    img1.float(), img2.float(), 
                    sum1.float(), sum2.float(),
                    name=f"Epoch {epoch+1}, Batch {idx}"
                )
            
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.6f}',
            'pred_range': f'[{result.min():.2f}, {result.max():.2f}]',
            'tgt_range': f'[{ref_HDR.min():.2f}, {ref_HDR.max():.2f}]'
        })
        
        # Tensorboard logging
        if (idx + 1) % 10 == 0:
            global_step = epoch * len(train_dataloader) + idx
            writer.add_scalar('Train/BatchLoss', loss.item() * accumulation_steps, global_step)
            writer.add_scalar('Train/PredMean', result.mean().item(), global_step)
            writer.add_scalar('Train/PredStd', result.std().item(), global_step)
            writer.add_scalar('Train/TargetMean', ref_HDR.mean().item(), global_step)
    
    # Handle remaining gradients
    if (idx + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    
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
        
        torch.cuda.synchronize()  # ensure all kernels finished
        torch.cuda.empty_cache() # release cached GPU memory
        gc.collect()
        
        # Evaluate
        eval_loss = 0
        with torch.no_grad():
            eval_loss = eval_one_epoch(epoch)
            logger.log(f"Eval Loss: {eval_loss:.8f}")
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
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
    
    # eval_loss = eval_one_epoch(epoch)
    # logger.log(f"Eval Loss: {eval_loss:.8f}")
        
    # Training complete
    total_time = time.time() - training_start_time
    logger.log("\n" + "="*80)
    logger.log("Training Completed!")
    logger.log(f"Total training time: {total_time/3600:.2f} hours")
    logger.log(f"Best validation loss: {best_loss:.8f}")
    logger.log("="*80)
    
    writer.close()
    
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    


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