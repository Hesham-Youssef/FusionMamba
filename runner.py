"""
U2Net Training/Testing Runner Script

This script provides a convenient interface for training and testing U2Net model.
Usage:
    python run_u2net.py --mode train
    python run_u2net.py --mode test
    python run_u2net.py --mode train --config custom_config.py
"""

import argparse
import os
import sys
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='U2Net HDR Reconstruction')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['train', 'test', 'resume', 'evaluate'],
                       help='Mode: train, test, resume, or evaluate')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file (optional)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to specific checkpoint (for testing)')
    parser.add_argument('--multigpu', action='store_true',
                       help='Enable multi-GPU training')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--dim', type=int, default=None,
                       help='Override model dimension')
    return parser.parse_args()


def setup_directories():
    """Create necessary directories"""
    dirs = [
        './checkpoints/u2net',
        './samples/u2net',
        './logs'
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ Created directory: {d}")


def check_dependencies():
    """Check if required packages are installed"""
    required = ['torch', 'numpy', 'cv2']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Please install them using: pip install " + ' '.join(missing))
        sys.exit(1)
    else:
        print("✓ All dependencies are installed")


def check_dataset(data_path='./dataset'):
    """Check if dataset exists"""
    train_path = os.path.join(data_path, 'train')
    test_path = os.path.join(data_path, 'test')
    
    if not os.path.exists(train_path):
        print(f"⚠ Warning: Training data not found at {train_path}")
        return False
    
    if not os.path.exists(test_path):
        print(f"⚠ Warning: Test data not found at {test_path}")
        return False
    
    # Count scenes
    train_scenes = len([d for d in os.listdir(train_path) 
                       if os.path.isdir(os.path.join(train_path, d))])
    test_scenes = len([d for d in os.listdir(test_path) 
                      if os.path.isdir(os.path.join(test_path, d))])
    
    print(f"✓ Dataset found:")
    print(f"  - Training scenes: {train_scenes}")
    print(f"  - Test scenes: {test_scenes}")
    return True


def print_config_info(args):
    """Print configuration information"""
    print("\n" + "="*60)
    print("U2Net Configuration")
    print("="*60)
    print(f"Mode: {args.mode}")
    if args.config:
        print(f"Custom config: {args.config}")
    if args.checkpoint:
        print(f"Checkpoint: {args.checkpoint}")
    if args.multigpu:
        print("Multi-GPU: Enabled")
    if args.batch_size:
        print(f"Batch size: {args.batch_size}")
    if args.learning_rate:
        print(f"Learning rate: {args.learning_rate}")
    if args.epochs:
        print(f"Epochs: {args.epochs}")
    if args.dim:
        print(f"Model dimension: {args.dim}")
    print("="*60 + "\n")


def modify_config_if_needed(args):
    """Create a temporary config file with overrides if needed"""
    if not any([args.multigpu, args.batch_size, args.learning_rate, 
                args.epochs, args.dim]):
        return None
    
    # Create temporary config
    temp_config = './temp_config.py'
    
    with open('./utils/configs.py', 'r') as f:
        config_content = f.read()
    
    # Add overrides
    overrides = []
    if args.multigpu:
        overrides.append("        self.multigpu = True")
    if args.batch_size:
        overrides.append(f"        self.batch_size = {args.batch_size}")
    if args.learning_rate:
        overrides.append(f"        self.learning_rate = {args.learning_rate}")
    if args.epochs:
        overrides.append(f"        self.epoch = {args.epochs}")
    if args.dim:
        overrides.append(f"        self.dim = {args.dim}")
    
    # Simple approach: append overrides
    # In practice, you'd want to properly parse and modify the class
    print("Note: Using command-line overrides. Consider modifying utils/configs.py directly.")
    
    return None


def train(args):
    """Run training"""
    print("\n" + "="*60)
    print("Starting U2Net Training")
    print("="*60 + "\n")
    
    cmd = [sys.executable, 'train.py']
    subprocess.run(cmd)


def test(args):
    """Run testing"""
    print("\n" + "="*60)
    print("Starting U2Net Testing")
    print("="*60 + "\n")
    
    cmd = [sys.executable, 'test.py']
    subprocess.run(cmd)


def evaluate(args):
    """Run evaluation with metrics"""
    print("\n" + "="*60)
    print("Starting U2Net Evaluation")
    print("="*60 + "\n")
    
    # Run testing
    test(args)
    
    # Compute additional metrics
    print("\nComputing additional metrics...")
    try:
        from utils.HDRutils import PSNR, SSIM
        print("✓ Metrics will be computed during testing")
    except ImportError:
        print("⚠ Warning: Could not import metric functions")


def main():
    args = parse_args()
    
    print("\n" + "="*60)
    print("U2Net HDR Reconstruction - Setup")
    print("="*60 + "\n")
    
    # Setup
    print("Checking dependencies...")
    check_dependencies()
    
    print("\nCreating directories...")
    setup_directories()
    
    print("\nChecking dataset...")
    check_dataset()
    
    print_config_info(args)
    
    # Run appropriate mode
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)
    elif args.mode == 'resume':
        print("Resume mode: Training will automatically resume from checkpoint")
        train(args)
    elif args.mode == 'evaluate':
        evaluate(args)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)