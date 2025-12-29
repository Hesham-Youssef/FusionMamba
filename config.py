"""
U2Net Configuration - Optimized for HDR Reconstruction
"""

class Configs:
    def __init__(self):
        # Data paths
        self.data_path = './dataset'
        self.checkpoint_dir = './checkpoints/u2net'
        self.sample_dir = './samples/u2net'
        
        # Model architecture parameters
        self.dim = 32  # Base dimension
        self.c_dim = 3  # RGB channels
        self.num_shots = 3
        
        # Training parameters - CRITICAL FIXES
        self.batch_size = 64  # REDUCED from 110 for better gradient stability
        self.learning_rate = 5e-5  # LOWERED from 2e-4 to prevent overshooting
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epoch = 150
        
        # Gradient accumulation for larger effective batch size
        self.accumulation_steps = 2  # Effective batch = 64 * 2 = 128
        
        # Data processing
        self.patch_size = [64, 64]
        self.image_size = [64, 64]
        self.patch_stride = 32
        
        # Hardware settings
        self.multigpu = False
        self.num_workers = 12
        
        # Reproducibility
        self.seed = 42
        
        # Loss function weights - NEW
        self.loss_config = {
            'type': 'improved',  # Use improved loss with SSIM + gradient
            'l1_weight': 1.0,
            'gradient_weight': 0.3,
            'range_penalty_weight': 3.0,  # HIGH penalty for range violations
            'ssim_weight': 0.0,  # Start with 0, can increase later if stable
        }
        
        # Training optimizations
        self.use_mixed_precision = True  # AMP for faster training
        self.gradient_clip_norm = 1.0  # Gradient clipping
        
    def __str__(self):
        """Print configuration"""
        config_str = "U2Net Configuration:\n"
        config_str += "="*60 + "\n"
        for key, value in self.__dict__.items():
            if isinstance(value, dict):
                config_str += f"{key:20s}:\n"
                for k, v in value.items():
                    config_str += f"  {k:18s}: {v}\n"
            else:
                config_str += f"{key:20s}: {value}\n"
        config_str += "="*60
        return config_str


if __name__ == '__main__':
    configs = Configs()
    print(configs)
    
    print("\n🔧 Key Changes from Default:")
    print("-" * 60)
    print("1. ✅ Learning rate: 2e-4 → 5e-5 (prevents overshooting)")
    print("2. ✅ Batch size: 110 → 64 (better gradient quality)")
    print("3. ✅ Added accumulation_steps=2 (effective batch=128)")
    print("4. ✅ Added loss_config with range penalty")
    print("5. ✅ High range_penalty_weight=3.0 (enforces [-1,1] bounds)")