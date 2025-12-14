import os
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.tools import ERGAS
from torchinfo import summary
import torch.backends.cudnn as cudnn
from model.u2net import U2Net as Net
from torch.utils.data import DataLoader
from utils.hdr_load_train_data import HDRDatasetTiles
import cv2

SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
cudnn.benchmark = True


def save_batch_debug(gt, ldr1, ldr2, sr, epoch, iteration):
    """Save first batch for debugging purposes."""
    out_dir = f"debug/epoch_{epoch}"
    os.makedirs(out_dir, exist_ok=True)

    def to_hdr(x):
        """Convert tensor to HDR numpy array for saving."""
        img = x[0].detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
        if img.shape[2] == 3:
            img = img[..., ::-1]  # RGB -> BGR for OpenCV
        return img

    # Save HDR files
    cv2.imwrite(os.path.join(out_dir, f"gt_{iteration}.hdr"), to_hdr(gt))
    cv2.imwrite(os.path.join(out_dir, f"ldr1_{iteration}.hdr"), to_hdr(ldr1))
    cv2.imwrite(os.path.join(out_dir, f"ldr2_{iteration}.hdr"), to_hdr(ldr2))
    cv2.imwrite(os.path.join(out_dir, f"sr_{iteration}.hdr"), to_hdr(sr))


def save_checkpoint(args, model, optimizer, scheduler, epoch):
    """Save model checkpoint."""
    os.makedirs(args.weight_dir, exist_ok=True)
    save_path = os.path.join(args.weight_dir, f"epoch_{epoch}.pth")

    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict()
    }, save_path)

    print(f"Checkpoint saved: {save_path}")


def prepare_training_data(args):
    """Prepare training and validation data loaders."""
    # Use tile dimensions from args (H and W are the tile dimensions)
    tile_h = args.H
    tile_w = args.W
    stride_h = args.stride_H
    stride_w = args.stride_W

    print(f"Dataset configuration:")
    print(f"  Tile size: {tile_h}x{tile_w}")
    print(f"  Stride: {stride_h}x{stride_w}")

    # Training dataset
    train_set = HDRDatasetTiles(
        data_dir=args.train_data_path,
        tile_h=tile_h,
        tile_w=tile_w,
        stride_h=stride_h,
        stride_w=stride_w
    )

    # Validation dataset
    validate_set = HDRDatasetTiles(
        data_dir=args.val_data_path,
        tile_h=tile_h,
        tile_w=tile_w,
        stride_h=stride_h,
        stride_w=stride_w
    )

    print(f"Training tiles: {len(train_set)}")
    print(f"Validation tiles: {len(validate_set)}")

    # DataLoader settings optimized for GPU training
    training_data_loader = DataLoader(
        dataset=train_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,  # Enable for faster GPU transfer
        drop_last=True,   # Drop last incomplete batch for consistent batch sizes
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    validate_data_loader = DataLoader(
        dataset=validate_set,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    
    return training_data_loader, validate_data_loader


def train(args, training_data_loader, validate_data_loader):
    """Main training loop."""
    # Initialize model
    model = Net(
        args.channels,
        args.first_channels,
        args.second_channels,
        args.H,
        args.W,
        args.ratio
    ).to(args.device)
    
    # Model expects all inputs to be the same size (tile_h x tile_w)
    # The ratio parameter is used internally by the model for processing
    print("\nModel Architecture:")
    summary(
        model,
        input_size=[
            (args.batch_size, 3, args.H, args.W),  # ldr1
            (args.batch_size, 3, args.H, args.W),  # ldr2
            (args.batch_size, 3, args.H, args.W),  # sum1
            (args.batch_size, 3, args.H, args.W)   # sum2
        ],
        dtypes=[torch.float, torch.float, torch.float, torch.float]
    )

    # Loss functions
    if args.use_ergas:
        criterion_l1 = nn.L1Loss().to(args.device)
        criterion_ergas = ERGAS(args.ratio).to(args.device)
    else:
        criterion = nn.L1Loss().to(args.device)

    # Optimizer and scheduler
    start_epoch = 1
    end_epoch = args.epoch
    num_epochs_to_run = end_epoch - start_epoch + 1
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=len(training_data_loader) * num_epochs_to_run,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # Resume from checkpoint if provided
    if args.resume_path is not None and os.path.isfile(args.resume_path):
        print(f"\nLoading checkpoint: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=args.device)

        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state'])

        start_epoch = checkpoint['epoch'] + 1
        end_epoch = start_epoch + num_epochs_to_run - 1
        print(f"Resumed training from epoch {start_epoch}")

    print(f"\nTraining from epoch {start_epoch} to {end_epoch}")
    print("=" * 80)
    
    t_start = time.time()
    
    # Training loop
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()
        epoch_train_loss = []
        epoch_train_loss_l1 = []
        epoch_train_loss_ergas = []
        
        for iteration, batch in enumerate(training_data_loader, 1):
            gt, ldr1, ldr2, sum1, sum2 = (
                batch[0].to(args.device),
                batch[1].to(args.device),
                batch[2].to(args.device),
                batch[3].to(args.device),
                batch[4].to(args.device)
            )
            
            optimizer.zero_grad()
            sr = model(ldr1, ldr2, sum1, sum2)
            
            # Optional: Save first batch of first epoch for debugging
            if args.save_debug and epoch == start_epoch and iteration == 1:
                save_batch_debug(gt, ldr1, ldr2, sr, epoch, iteration)

            # Compute loss
            if args.use_ergas:
                loss_l1 = criterion_l1(sr, gt)
                loss_ergas = criterion_ergas(sr, gt)
                loss = loss_l1 + args.ergas_hp * loss_ergas
                epoch_train_loss_l1.append(loss_l1.item())
                epoch_train_loss_ergas.append(loss_ergas.item())
            else:
                loss = criterion(sr, gt)
            
            epoch_train_loss.append(loss.item())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Print progress
            if iteration % args.print_freq == 0:
                print(f'Epoch: {epoch}/{end_epoch} | '
                      f'Iter: {iteration}/{len(training_data_loader)} | '
                      f'Loss: {loss.item():.6f} | '
                      f'LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Epoch summary
        t_loss = np.nanmean(np.array(epoch_train_loss))
        if args.use_ergas:
            print(f'\nEpoch {epoch}/{end_epoch} Training Summary:')
            print(f'  Total Loss: {t_loss:.6f}')
            print(f'  L1 Loss: {np.nanmean(np.array(epoch_train_loss_l1)):.6f}')
            print(f'  ERGAS Loss: {np.nanmean(np.array(epoch_train_loss_ergas)):.6f}')
        else:
            print(f'\nEpoch {epoch}/{end_epoch} Training Loss: {t_loss:.6f}')

        # Validation
        if epoch % args.val_freq == 0:
            model.eval()
            epoch_val_loss = []
            epoch_val_loss_l1 = []
            epoch_val_loss_ergas = []
            
            with torch.no_grad():
                for iteration, batch in enumerate(validate_data_loader, 1):
                    gt, ldr1, ldr2, sum1, sum2 = (
                        batch[0].to(args.device),
                        batch[1].to(args.device),
                        batch[2].to(args.device),
                        batch[3].to(args.device),
                        batch[4].to(args.device)
                    )
                    
                    sr = model(ldr1, ldr2, sum1, sum2)
                    
                    if args.use_ergas:
                        loss_l1 = criterion_l1(sr, gt)
                        loss_ergas = criterion_ergas(sr, gt)
                        loss = loss_l1 + args.ergas_hp * loss_ergas
                        epoch_val_loss_l1.append(loss_l1.item())
                        epoch_val_loss_ergas.append(loss_ergas.item())
                    else:
                        loss = criterion(sr, gt)
                    
                    epoch_val_loss.append(loss.item())
            
            v_loss = np.nanmean(np.array(epoch_val_loss))
            t_end = time.time()
            
            print(f'\n{"="*80}')
            print(f'Validation Results:')
            if args.use_ergas:
                print(f'  Total Loss: {v_loss:.6f}')
                print(f'  L1 Loss: {np.nanmean(np.array(epoch_val_loss_l1)):.6f}')
                print(f'  ERGAS Loss: {np.nanmean(np.array(epoch_val_loss_ergas)):.6f}')
            else:
                print(f'  Loss: {v_loss:.6f}')
            print(f'Time elapsed: {(t_end - t_start):.2f}s')
            print(f'{"="*80}\n')
            
            t_start = time.time()

        # Save checkpoint
        if epoch % args.ckpt == 0:
            save_checkpoint(args, model, optimizer, lr_scheduler, epoch)
    
    # Save final checkpoint
    save_checkpoint(args, model, optimizer, lr_scheduler, end_epoch)
    print("\nTraining completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HDR Image Reconstruction Training')
    
    # Model architecture
    parser.add_argument('--ratio', type=int, default=1, 
                        help='Internal processing ratio used by model')
    parser.add_argument('--H', type=int, default=64, 
                        help='Tile height')
    parser.add_argument('--W', type=int, default=64, 
                        help='Tile width')
    parser.add_argument('--stride_H', type=int, default=32, 
                        help='Vertical stride for tile extraction')
    parser.add_argument('--stride_W', type=int, default=32, 
                        help='Horizontal stride for tile extraction')
    parser.add_argument('--channels', type=int, default=32, 
                        help='Number of feature channels')
    parser.add_argument('--first_channels', type=int, default=3, 
                        help='Number of spatial channels')
    parser.add_argument('--second_channels', type=int, default=3, 
                        help='Number of spectral channels')
    
    # Loss function
    parser.add_argument('--use_ergas', action='store_true', 
                        help='Use ERGAS loss in addition to L1 loss')
    parser.add_argument('--ergas_hp', type=float, default=1e-4, 
                        help='Weight for ERGAS loss')
    
    # Training parameters
    parser.add_argument('--epoch', type=int, default=500, 
                        help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=20, 
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, 
                        help='Initial learning rate')
    parser.add_argument('--num_workers', type=int, default=4, 
                        help='Number of data loader workers')
    
    # Checkpointing and validation
    parser.add_argument('--ckpt', type=int, default=20, 
                        help='Save checkpoint every N epochs')
    parser.add_argument('--val_freq', type=int, default=1, 
                        help='Run validation every N epochs')
    parser.add_argument('--print_freq', type=int, default=10, 
                        help='Print training stats every N iterations')
    parser.add_argument('--save_debug', action='store_true', 
                        help='Save debug images from first batch')
    
    # Paths
    parser.add_argument('--device', type=str, default='cuda', 
                        choices=['cuda', 'cpu'])
    parser.add_argument('--train_data_path', type=str, required=True,
                        help='Path to training dataset')
    parser.add_argument('--val_data_path', type=str, required=True,
                        help='Path to validation dataset')
    parser.add_argument('--weight_dir', type=str, default='weights/', 
                        help='Directory to save checkpoints')
    parser.add_argument('--resume_path', type=str, default=None, 
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.H <= 0 or args.W <= 0:
        raise ValueError("Tile dimensions (H, W) must be positive")
    if args.stride_H <= 0 or args.stride_W <= 0:
        raise ValueError("Stride dimensions must be positive")
    if args.stride_H > args.H or args.stride_W > args.W:
        print("Warning: Stride larger than tile size will create gaps in coverage")
    
    print("\nTraining Configuration:")
    print(f"  Device: {args.device}")
    print(f"  Tile size: {args.H}x{args.W}")
    print(f"  Stride: {args.stride_H}x{args.stride_W}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epoch}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Use ERGAS: {args.use_ergas}")
    if args.use_ergas:
        print(f"  ERGAS weight: {args.ergas_hp}")
    print()

    training_data_loader, validate_data_loader = prepare_training_data(args)
    train(args, training_data_loader, validate_data_loader)