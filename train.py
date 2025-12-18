import os
import time
import csv
from pathlib import Path
import argparse

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from torchinfo import summary
from torch.utils.data import DataLoader

from utils.tools import compute_hdr_loss
from model.u2net import U2Net as Net
from utils.hdr_load_train_data import HDRDatasetMaxPerf
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


# -----------------------------
# CSV split helper
# -----------------------------
def read_split_csv(csv_path):
    """
    Read a CSV/TSV with columns 'scene' and 'split' and return a dict mapping split -> set(scenes).
    Accepts comma or tab delimiters.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")

    # try comma first, then tab
    for delimiter in [',', '\t']:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter, skipinitialspace=True)
            if reader.fieldnames and 'scene' in reader.fieldnames and 'split' in reader.fieldnames:
                splits = {}
                f.seek(0)
                reader = csv.DictReader(f, delimiter=delimiter, skipinitialspace=True)
                for row in reader:
                    scene = row.get('scene')
                    sp = row.get('split')
                    if scene is None or sp is None:
                        continue
                    scene = scene.strip()
                    sp = sp.strip()
                    splits.setdefault(sp, set()).add(scene)
                return splits
    raise RuntimeError("CSV must contain 'scene' and 'split' columns (tab or comma separated).")


# -----------------------------
# Prepare data loaders
# -----------------------------
def prepare_training_data(args):
    """Prepare training and validation data loaders with optimized features.
    Supports two modes:
    1) CSV split mode: pass --split_csv and --data_dir, script reads CSV and filters scenes.
    2) Classic mode: pass --train_data_path and --val_data_path (backwards compatible).
    """
    # Use tile dimensions from args (H and W are the tile dimensions)
    tile_h = args.H
    tile_w = args.W
    stride_h = args.stride_H
    stride_w = args.stride_W

    print(f"\n{'='*80}")
    print("Dataset Configuration:")
    print(f"  Tile size: {tile_h}x{tile_w}")
    print(f"  Stride: {stride_h}x{stride_w}")
    print(f"\nOptimization Features:")
    print(f"  Parallel load threads: {args.num_load_threads}")
    print(f"  Disk cache: {args.use_disk_cache}")
    if args.use_disk_cache:
        print(f"  Disk cache dir: {args.disk_cache_dir}")
    print(f"  Max cached pairs: {args.max_cached_pairs}")
    print(f"  Smart sampler: {args.use_smart_sampler}")
    print(f"{'='*80}\n")

    if not args.data_dir:
        # fallback to train_data_path if provided
        if args.train_data_path:
            data_dir = args.train_data_path
        elif args.val_data_path:
            data_dir = args.val_data_path
        else:
            raise ValueError("When using --split_csv you must provide --data_dir or --train_data_path/--val_data_path")
    else:
        data_dir = args.data_dir

    print(f"Reading split CSV: {args.split_csv}")
    splits = read_split_csv(args.split_csv)
    train_scenes = splits.get('train', set())
    val_scenes = splits.get('val', set())

    available = set(p.name for p in Path(data_dir).iterdir() if p.is_dir())
    missing_train = train_scenes - available
    missing_val = val_scenes - available
    if missing_train:
        print(f"Warning: {len(missing_train)} train scenes from CSV not found in {data_dir}. They will be ignored.")
    if missing_val:
        print(f"Warning: {len(missing_val)} val scenes from CSV not found in {data_dir}. They will be ignored.")

    train_scenes = sorted(list(train_scenes & available))
    val_scenes = sorted(list(val_scenes & available))

    print(f"Using {len(train_scenes)} scenes for training, {len(val_scenes)} for validation (from CSV).")

    train_set = HDRDatasetMaxPerf(
        data_dir=data_dir,
        tile_h=tile_h, tile_w=tile_w,
        stride_h=stride_h, stride_w=stride_w,
        split='train',
        split_scenes=train_scenes,
        max_cached_pairs=args.max_cached_pairs,
        num_load_threads=args.num_load_threads,
        use_disk_cache=args.use_disk_cache,
        disk_cache_dir=args.disk_cache_dir if args.use_disk_cache else None,
        summary_only=args.summary_only,
        disk_max_size_bytes=args.disk_max_size_bytes,
    )

    print(f"Training tiles: {len(train_set)}")

    print("Using smart sampler for better cache locality")
    train_sampler = train_set.get_smart_sampler(args.batch_size, shuffle=True)
    training_data_loader = DataLoader(
        dataset=train_set,
        batch_sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=False if args.num_workers > 0 else False,
        prefetch_factor=1 if args.num_workers > 0 else None
    )
    
    return training_data_loader, train_set


def train(args, training_data_loader, train_set):
    """Main training loop with fixed HDR loss."""

    # Initialize model
    model = Net(
        args.channels,
        args.first_channels,
        args.second_channels,
        args.H,
        args.W
    ).to(args.device)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    # Print model summary
    print("\nModel Architecture:")
    summary(
        raw_model,
        input_size=[
            (args.batch_size, 3, args.H, args.W),  # ldr1
            (args.batch_size, 3, args.H, args.W),  # ldr2
            (args.batch_size, 3, args.H, args.W),  # sum1
            (args.batch_size, 3, args.H, args.W)   # sum2
        ],
        dtypes=[torch.float, torch.float, torch.float, torch.float]
    )
    
    # Optimizer and LR scheduler
    optimizer = torch.optim.AdamW([
        {
        "params": [p for n, p in raw_model.named_parameters()
                   if "scale_net" not in n and "skip_scale_net" not in n],
        "lr": args.lr
        },
        {"params": model.scale_net.parameters(), "lr": args.lr * 0.1},
        {"params": model.skip_scale_net.parameters(), "lr": args.lr * 0.1},
    ])
    num_epochs_to_run = args.epoch
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=len(training_data_loader) * num_epochs_to_run,
        pct_start=0.3,
        anneal_strategy='cos'
    )

    # Resume from checkpoint
    start_epoch = 1
    if args.resume_path and os.path.isfile(args.resume_path):
        print(f"\nLoading checkpoint: {args.resume_path}")
        checkpoint = torch.load(args.resume_path, map_location=args.device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed training from epoch {start_epoch}")

    print(f"\nTraining from epoch {start_epoch} to {args.epoch}")
    print("=" * 80)
    t_start = time.time()

    try:
        for epoch in range(start_epoch, args.epoch + 1):
            model.train()

            for iteration, batch in enumerate(training_data_loader, 1):
                gt, ldr1, ldr2, sum1, sum2 = [x.to(args.device) for x in batch]

                optimizer.zero_grad()
                sr_log, sr_linear, adaptive_scale, skip_scale = model(ldr1, ldr2, sum1, sum2)

                target_log = torch.log1p(gt)
                
                loss = compute_hdr_loss(
                    sr_log, target_log, sr_linear, gt,
                    adaptive_scale, skip_scale,
                    epoch=(iteration + epoch * len(training_data_loader))//100
                )

                
                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()

                # Progress printing
                if iteration % args.print_freq == 0:
                    print(
                        f'Epoch {epoch}/{args.epoch} | '
                        f'Iter {iteration}/{len(training_data_loader)} | '
                        f'total_loss: {loss.item():.4f} | '
                        f'LR: {optimizer.param_groups[0]["lr"]:.2e}'
                    )
                    print(f'  sr_linear: min={sr_linear.min():.4f}, max={sr_linear.max():.4f}, mean={sr_linear.mean():.4f}')
                    print(f'  gt: min={gt.min():.4f}, max={gt.max():.4f}, mean={gt.mean():.4f}')
                                
                    print(f'{"-"*80}\n')

                # Checkpoint every 200 iterations
                if iteration % 200 == 0:
                    save_checkpoint(args, model, optimizer, lr_scheduler, iteration + epoch)
                    print(f"\nCheckpoint saved at iteration {iteration + epoch}!")


            print(f'\n{"="*80}')
            print(f'Time elapsed: {time.time() - t_start:.2f}s')
            print(f'{"="*80}\n')
            t_start = time.time()

            save_checkpoint(args, raw_model, optimizer, lr_scheduler, epoch)

        # Final checkpoint
        # save_checkpoint(args, model, optimizer, lr_scheduler, args.epoch)
        print("\nTraining completed!")

    finally:
        # Cleanup
        print("\nCleaning up datasets...")
        train_set.cleanup()
        print("Cleanup completed.")


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

    # DataLoader optimization features
    parser.add_argument('--summary_only', type=bool, default=True)
    parser.add_argument('--disk_max_size_bytes', type=int, default=5 * (1024 ** 3))
    parser.add_argument('--num_load_threads', type=int, default=4,
                        help='Number of parallel image loading threads')
    parser.add_argument('--use_disk_cache', action='store_true',
                        help='Cache preprocessed tiles to disk')
    parser.add_argument('--disk_cache_dir', type=str, default='./tile_cache',
                        help='Directory for disk cache')
    parser.add_argument('--max_cached_pairs', type=int, default=16,
                        help='Maximum number of pairs to keep in memory cache')
    parser.add_argument('--use_smart_sampler', action='store_true',
                        help='Use smart sampler for better cache locality')

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
    parser.add_argument('--train_data_path', type=str, default=None,
                        help='Path to training dataset (use when not using --split_csv)')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Path to validation dataset (use when not using --split_csv)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to full dataset (use with --split_csv)')
    parser.add_argument('--split_csv', type=str, default=None,
                        help="Path to CSV/TSV file with columns 'scene' and 'split' (use with --data_dir)")
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

    print("\n" + "="*80)
    print("Training Configuration:")
    print("="*80)
    print(f"  Device: {args.device}")
    print(f"  Tile size: {args.H}x{args.W}")
    print(f"  Stride: {args.stride_H}x{args.stride_W}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epoch}")
    print(f"  Learning rate: {args.lr}")
    print("="*80)

    training_data_loader, train_set = prepare_training_data(args)

    train(args, training_data_loader, train_set)
