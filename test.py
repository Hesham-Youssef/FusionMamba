import os
import torch
from torch import optim
from torch.utils.data import DataLoader
from utils.tools import *
from model.u2net import U2Net
import numpy as np
from utils.hdr_load_train_data import *
from config import Configs

# Get configurations
configs = Configs()

# Load test dataset
print("Loading test dataset...")
test_dataset = U2NetTestDataset(configs=configs)
test_dataloader = DataLoader(
    test_dataset, 
    batch_size=1, 
    shuffle=False
)

# Build U2Net model
print("Building U2Net model...")
dim = configs.dim if hasattr(configs, 'dim') else 32
img1_dim = configs.c_dim * 2  # LDR channels
img2_dim = configs.c_dim * 2  # HDR channels
H = configs.patch_size[0] if hasattr(configs, 'patch_size') else 64
W = configs.patch_size[1] if hasattr(configs, 'patch_size') else 64

model = U2Net(dim=dim, img1_dim=img1_dim, img2_dim=img2_dim, H=H, W=W)

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

# Define optimizer (needed for checkpoint loading)
optimizer = optim.Adam(
    model.parameters(), 
    betas=(configs.beta1, configs.beta2), 
    lr=configs.learning_rate
)

# Define Criterion
criterion = HDRLoss()

# Load checkpoint
checkpoint_file = os.path.join(configs.checkpoint_dir, 'checkpoint.tar')
best_checkpoint_file = os.path.join(configs.checkpoint_dir, 'best_checkpoint.tar')

# Try to load best checkpoint first, fallback to latest checkpoint
if os.path.isfile(best_checkpoint_file):
    print(f"Loading best checkpoint from {best_checkpoint_file}")
    checkpoint = torch.load(best_checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best checkpoint (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.8f})")
elif os.path.isfile(checkpoint_file):
    print(f"Loading checkpoint from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint (epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.8f})")
else:
    raise FileNotFoundError(f'No checkpoint files found in {configs.checkpoint_dir}')

# Enable DataParallel if needed
if configs.multigpu is True:
    model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")


def test_one_epoch():
    """Test the model on test dataset"""
    model.eval()
    mean_loss = 0
    count = 0
    losses = []
    
    print(f"\nTesting on {len(test_dataloader)} samples...")
    print("="*60)
    
    for idx, data in enumerate(test_dataloader):
        # Unpack data
        sample_path, img1_ldr, img2_ldr, img1_hdr, img2_hdr, sum1, sum2, ref_HDR = data
        sample_path = sample_path[0]
        
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
        
        # Forward pass (no gradient)
        with torch.no_grad():
            res = model(img1, img2, sum1, sum2)
            loss = criterion(tonemap(res), tonemap(ref_HDR))
        
        # Save output
        dump_sample(sample_path, res.cpu().detach().numpy())
        
        # Track metrics
        loss_val = loss.item()
        losses.append(loss_val)
        mean_loss += loss_val
        count += 1
        
        print(f'Test Batch [{idx + 1}/{len(test_dataloader)}] - '
              f'Sample: {os.path.basename(sample_path)} - '
              f'Loss: {loss_val:.8f}')

    mean_loss = mean_loss / count
    std_loss = np.std(losses)
    
    return mean_loss, std_loss, losses


def test():
    """Main testing function"""
    print("\n" + "="*60)
    print("Starting U2Net Testing")
    print("="*60)
    
    mean_loss, std_loss, losses = test_one_epoch()
    
    print("\n" + "="*60)
    print("Testing Results")
    print("="*60)
    print(f"Mean Test Loss: {mean_loss:.8f}")
    print(f"Std Test Loss:  {std_loss:.8f}")
    print(f"Min Test Loss:  {min(losses):.8f}")
    print(f"Max Test Loss:  {max(losses):.8f}")
    print("="*60)
    
    # Save results to file
    results_file = os.path.join(configs.sample_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("U2Net Testing Results\n")
        f.write("="*60 + "\n")
        f.write(f"Mean Test Loss: {mean_loss:.8f}\n")
        f.write(f"Std Test Loss:  {std_loss:.8f}\n")
        f.write(f"Min Test Loss:  {min(losses):.8f}\n")
        f.write(f"Max Test Loss:  {max(losses):.8f}\n")
        f.write("\nPer-sample losses:\n")
        for i, loss in enumerate(losses):
            f.write(f"Sample {i+1}: {loss:.8f}\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    # Create sample directory if it doesn't exist
    os.makedirs(configs.sample_dir, exist_ok=True)
    
    test()
    print("\nTesting completed!")