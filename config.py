"""
U2Net Configuration Template

This file contains the configuration parameters needed for U2Net training and testing.
Copy this template and modify according to your dataset and requirements.
"""

class Configs:
    def __init__(self):
        # Data paths
        self.data_path = './dataset'  # Root path to dataset
        self.checkpoint_dir = './checkpoints/u2net'  # Where to save model checkpoints
        self.sample_dir = './samples/u2net'  # Where to save test outputs
        
        # Model architecture parameters
        self.dim = 32  # Base dimension for U2Net (channels in first layer)
        self.c_dim = 3  # Color channels (RGB = 3)
        self.num_shots = 3  # Number of exposure images
        
        # Training parameters
        self.batch_size = 8  # Batch size for training
        self.learning_rate = 2e-4  # Initial learning rate
        self.beta1 = 0.9  # Adam optimizer beta1
        self.beta2 = 0.999  # Adam optimizer beta2
        self.epoch = 15  # Total number of training epochs
        
        # Data processing
        self.patch_size = [64, 64]  # Size of training patches [height, width]
        self.image_size = [64, 64]  # Size to resize images to [height, width]
        self.patch_stride = 32  # Stride for patch extraction (overlap = patch_size - stride)
        
        # Hardware settings
        self.multigpu = False  # Use multiple GPUs (DataParallel)
        self.num_workers = 12  # Number of data loading workers
        
        # Reproducibility
        self.seed = 42  # Random seed for reproducibility
        
    def __str__(self):
        """Print configuration"""
        config_str = "U2Net Configuration:\n"
        config_str += "="*60 + "\n"
        for key, value in self.__dict__.items():
            config_str += f"{key:20s}: {value}\n"
        config_str += "="*60
        return config_str


# Example usage:
if __name__ == '__main__':
    configs = Configs()
    print(configs)
    
    # You can modify specific parameters like this:
    # configs.batch_size = 16
    # configs.learning_rate = 1e-4
    # configs.dim = 64  # Use more channels for larger model
    
    # Important notes:
    print("\nImportant Configuration Notes:")
    print("-" * 60)
    print("1. patch_size: Should match the H, W parameters used in U2Net model initialization")
    print("   - Training uses patches of this size extracted from full images")
    print("   - Must be divisible by 8 for proper encoder-decoder operation")
    print("")
    print("2. patch_stride: Controls overlap between patches")
    print("   - Smaller stride = more overlap = more training samples")
    print("   - Typical: stride = patch_size // 2 for 50% overlap")
    print("")
    print("3. dim: Base channel dimension for U2Net")
    print("   - Larger dim = more capacity but slower training")
    print("   - dim=32: ~5M parameters, dim=64: ~20M parameters")
    print("")
    print("4. batch_size: Depends on GPU memory")
    print("   - 256x256 patches with dim=32: batch_size=8-16 on 12GB GPU")
    print("   - Reduce if you get out-of-memory errors")
    print("")
    print("5. Dataset structure expected:")
    print("   data_path/")
    print("   ├── train/")
    print("   │   ├── scene_001/")
    print("   │   │   ├── input_1_aligned.tif")
    print("   │   │   ├── input_2_aligned.tif")
    print("   │   │   ├── input_3_aligned.tif")
    print("   │   │   ├── input_exp.txt")
    print("   │   │   └── ref_hdr_aligned.hdr")
    print("   │   └── scene_002/...")
    print("   └── test/")
    print("       └── (same structure as train)")
    print("")
    print("6. U2Net expects inputs in this format:")
    print("   - img1: concatenated [LDR_image1, HDR_image1] (6 channels)")
    print("   - img2: concatenated [LDR_image2, HDR_image2] (6 channels)")
    print("   - sum1: average of all LDR images (3 channels)")
    print("   - sum2: average of all HDR images (3 channels)")
    print("   - Output: HDR image (3 channels)")