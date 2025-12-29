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
from torch.cuda.amp import autocast, GradScaler


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


def eval_one_epoch(epoch, save_outputs=True):
    model.eval()
    count = 0
    total_loss = 0
    total_loss_dict = {}

    patch_h, patch_w = configs.patch_size
    stride = patch_h // 2
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

            result = torch.zeros((1, ref_HDR.shape[1], H, W), 
                                device=device, dtype=torch.float32)
            weight_map = torch.zeros((1, 1, H, W), device=device)

            blend = make_blend_window(patch_h, patch_w, device, blend_type='hann')

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
            with torch.no_grad():
                result = model(img1, img2, sum1, sum2)

        # Compute loss with components
        loss, loss_dict = criterion(result, ref_HDR)
        total_loss += loss.item()
        
        # Accumulate loss components
        for key, value in loss_dict.items():
            total_loss_dict[key] = total_loss_dict.get(key, 0.0) + value
        
        count += 1
        
        # Diagnostics
        comprehensive_diagnostics(
            result, ref_HDR, 
            img1, img2, sum1, sum2,
            name=f"Eval - Epoch {epoch+1}, Scene {batch_idx}"
        )
        
        # Save outputs
        if save_outputs:
            save_diagnostic_images(result, ref_HDR, img1, img2, sample_path, batch_idx)
            from utils.tools import dump_sample
            dump_sample(sample_path, result.cpu().numpy())
            
            # Save visualization
            result_np = result.detach().cpu().numpy()[0]
            ref_HDR_np = ref_HDR.detach().cpu().numpy()[0]
            
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
    
    # Average loss components
    avg_loss_dict = {key: value / count for key, value in total_loss_dict.items()}
    return total_loss / max(count, 1), avg_loss_dict


def train_one_epoch(epoch):
    model.train()
    epoch_loss = 0.0
    epoch_loss_dict = {}
    
    scaler = GradScaler() if configs.use_mixed_precision else None
    accumulation_steps = configs.accumulation_steps
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_dataloader, desc=f'Epoch {epoch+1}/{configs.epoch} [Train]')
    
    for idx, data in enumerate(pbar):
        img1, img2, sum1, sum2, ref_HDR = data
        
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        sum1 = sum1.to(device, non_blocking=True)
        sum2 = sum2.to(device, non_blocking=True)
        ref_HDR = ref_HDR.to(device, non_blocking=True)
        
        # Forward pass
        if configs.use_mixed_precision:
            with autocast():
                result = model(img1, img2, sum1, sum2)
                loss, loss_dict = criterion(result, ref_HDR)
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps
        else:
            result = model(img1, img2, sum1, sum2)
            loss, loss_dict = criterion(result, ref_HDR)
            if accumulation_steps > 1:
                loss = loss / accumulation_steps
        
        # Accumulate loss components for logging
        for key, value in loss_dict.items():
            epoch_loss_dict[key] = epoch_loss_dict.get(key, 0.0) + value
        
        # Backward pass
        if configs.use_mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update weights
        if (idx + 1) % accumulation_steps == 0:
            if configs.use_mixed_precision:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=configs.gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=configs.gradient_clip_norm)
                optimizer.step()
            
            optimizer.zero_grad()
        
        epoch_loss += loss.item() * accumulation_steps
        
        # Detailed diagnostics every 10 batches
        if idx % 10 == 0:
            with torch.no_grad():
                comprehensive_diagnostics(
                    result.float(), ref_HDR.float(), 
                    img1.float(), img2.float(), 
                    sum1.float(), sum2.float(),
                    name=f"Epoch {epoch+1}, Batch {idx}"
                )
        
        # IMPROVED progress bar with loss components
        pbar.set_postfix({
            'loss': f'{loss.item() * accumulation_steps:.4f}',
            'l1': f'{loss_dict["l1"]:.4f}',
            'grad': f'{loss_dict["gradient"]:.4f}',
            'range': f'{loss_dict["range_penalty"]:.4f}',
            'pred': f'[{result.min():.2f},{result.max():.2f}]'
        })
        
        # Tensorboard logging
        if (idx + 1) % 10 == 0:
            global_step = epoch * len(train_dataloader) + idx
            writer.add_scalar('Train/BatchLoss', loss.item() * accumulation_steps, global_step)
            writer.add_scalar('Train/L1Loss', loss_dict['l1'], global_step)
            writer.add_scalar('Train/GradientLoss', loss_dict['gradient'], global_step)
            writer.add_scalar('Train/RangePenalty', loss_dict['range_penalty'], global_step)
            writer.add_scalar('Train/PredRange', result.max().item() - result.min().item(), global_step)
    
    # Handle remaining gradients
    if (idx + 1) % accumulation_steps != 0:
        if configs.use_mixed_precision:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=configs.gradient_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=configs.gradient_clip_norm)
            optimizer.step()
    
    avg_loss = epoch_loss / len(train_dataloader)
    avg_loss_dict = {key: value / len(train_dataloader) for key, value in epoch_loss_dict.items()}
    
    return avg_loss, avg_loss_dict


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

# Load datasets
logger.log("Loading training dataset...")
train_dataset = U2NetDataset(configs=configs)
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
test_dataset = U2NetTestDataset(configs=configs)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False,
    num_workers=configs.num_workers,
    prefetch_factor=4
)
logger.log(f"Test dataset loaded: {len(test_dataset)} scenes")

# Build U2Net model
logger.log("Building U2Net model...")
dim = configs.dim
img1_dim = configs.c_dim
img2_dim = configs.c_dim
H = configs.patch_size[0]
W = configs.patch_size[1]

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

    eval_loss, _ = eval_one_epoch(0)
    logger.log(f"Eval Loss: {eval_loss:.8f}")
else:
    logger.log("No checkpoint found, starting from scratch")

# Enable DataParallel if needed
if configs.multigpu is True:
    model = torch.nn.DataParallel(model)
    logger.log(f"Using {torch.cuda.device_count()} GPUs with DataParallel")


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
    logger.log(f"  Accumulation steps: {configs.accumulation_steps}")
    logger.log(f"  Effective batch size: {configs.batch_size * configs.accumulation_steps}")
    logger.log(f"  Learning rate: {configs.learning_rate}")
    logger.log(f"  Device: {device}")
    logger.log("="*80 + "\n")
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, configs.epoch):
        epoch_start_time = time.time()
        
        logger.log(f"\n{'='*80}")
        logger.log(f"Epoch [{epoch + 1}/{configs.epoch}]")
        logger.log(f"Learning rate: {lr_scheduler.get_last_lr()[0]:.8f}")
        logger.log(f"{'='*80}")
        
        # Train
        train_loss, train_loss_dict = train_one_epoch(epoch)
        logger.log(f"Train Loss: {train_loss:.8f}")
        logger.log(f"  L1: {train_loss_dict['l1']:.6f}")
        logger.log(f"  Gradient: {train_loss_dict['gradient']:.6f}")
        logger.log(f"  Range Penalty: {train_loss_dict['range_penalty']:.6f}")
        
        # Tensorboard logging
        writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        writer.add_scalar('Train/L1Loss', train_loss_dict['l1'], epoch)
        writer.add_scalar('Train/GradientLoss', train_loss_dict['gradient'], epoch)
        writer.add_scalar('Train/RangePenalty', train_loss_dict['range_penalty'], epoch)
        writer.add_scalar('Train/LearningRate', lr_scheduler.get_last_lr()[0], epoch)
        
        lr_scheduler.step()
        
        save_checkpoint(epoch, train_loss, best_loss, optimizer, model, lr_scheduler, False)
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        # Evaluate
        eval_loss = 0
        eval_loss_dict = {}
        if epoch % 10 == 0:
            with torch.no_grad():
                eval_loss, eval_loss_dict = eval_one_epoch(epoch)
                logger.log(f"Eval Loss: {eval_loss:.8f}")
                logger.log(f"  L1: {eval_loss_dict['l1']:.6f}")
                logger.log(f"  Gradient: {eval_loss_dict['gradient']:.6f}")
                logger.log(f"  Range Penalty: {eval_loss_dict['range_penalty']:.6f}")
                
                writer.add_scalar('Eval/Loss', eval_loss, epoch)
                writer.add_scalar('Eval/L1Loss', eval_loss_dict['l1'], epoch)
                writer.add_scalar('Eval/GradientLoss', eval_loss_dict['gradient'], epoch)
                writer.add_scalar('Eval/RangePenalty', eval_loss_dict['range_penalty'], epoch)
        
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        gc.collect()
        
        is_best = eval_loss < best_loss if eval_loss > 0 else False
        if is_best:
            logger.log(f"🎉 New best model! Loss improved from {best_loss:.8f} to {eval_loss:.8f}")
            best_loss = eval_loss
            save_checkpoint(epoch, eval_loss, best_loss, optimizer, model, lr_scheduler, True)
        
        epoch_time = time.time() - epoch_start_time
        logger.log(f"Epoch time: {epoch_time:.2f}s")
    
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