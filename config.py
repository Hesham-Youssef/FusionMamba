"""
U2Net Configuration - Fixed for Stable Training
"""

class Configs:
    def __init__(self):
        # Data paths
        self.data_path = './dataset'
        self.checkpoint_dir = './checkpoints/u2net'
        self.sample_dir = './samples/u2net'
        
        # Model architecture parameters
        self.dim = (32 * 3)  # Base dimension
        self.c_dim = 3  # RGB channels
        self.num_shots = 3
        
        self.batch_size = 30
        
        self.learning_rate = 1e-4      # Initial LR
        self.max_lr = 1e-3             # Max LR for scheduler
        self.min_lr = 1e-5             # Minimum LR
        
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epoch = 100
        
        self.lr_warmup_epochs = 5  # Longer warmup
        
        self.enable_shuffle = True
        
        # Data processing
        self.patch_size = [64, 64]
        self.image_size = [64, 64]
        self.patch_stride = 32
        
        # Hardware settings
        self.multigpu = False
        self.num_workers = 12
        
        # Reproducibility
        self.seed = 42
        
        # ✅ FIXED: Simplified loss weights
        self.MU = 5000.0  # Mu-law compression parameter
        
        # Training optimizations
        self.use_mixed_precision = True
        self.gradient_clip_norm = 100.0  # ✅ CRITICAL: Much lower clip threshold
        
        self.weight_decay = 1e-4  # Higher regularization
        
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