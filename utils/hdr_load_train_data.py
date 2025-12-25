import os
import cv2
from glob import glob
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset
from utils.tools import get_image, LDR2HDR_batch, LDR2HDR


def create_global_summary(full_image, target_size):
    """
    Compress full image into a summary of target_size.
    
    Args:
        full_image: (H, W, C) numpy array
        target_size: (h, w) tuple for output size
    
    Returns:
        summary: (h, w, C) downsampled image
    """
    h, w, c = full_image.shape
    target_h, target_w = target_size
    
    # Use area interpolation (best for downsampling)
    summary = cv2.resize(full_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return summary


class SummaryCache:
    """
    Disk-based cache for global summaries.
    Stores summaries as numpy files for fast loading.
    """
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, scene_name, exp_idx, mode='ldr'):
        """Get the cache file path for a specific summary."""
        return self.cache_dir / f"{scene_name}_exp{exp_idx}_{mode}.npy"
    
    def exists(self, scene_name, exp_idx, mode='ldr'):
        """Check if a cached summary exists."""
        return self.get_cache_path(scene_name, exp_idx, mode).exists()
    
    def save(self, scene_name, exp_idx, mode, data):
        """Save a summary to disk."""
        cache_path = self.get_cache_path(scene_name, exp_idx, mode)
        np.save(cache_path, data.astype(np.float32))
    
    def load(self, scene_name, exp_idx, mode='ldr'):
        """Load a summary from disk."""
        cache_path = self.get_cache_path(scene_name, exp_idx, mode)
        return np.load(cache_path)
    
    def build_scene_cache(self, scene_path, scene_name, input_paths, exposures, 
                          patch_size, num_shots):
        """
        Build cache for all exposures in a scene.
        This is called once during dataset initialization.
        """
        print(f"Building cache for scene: {scene_name}")
        
        for idx in range(num_shots):
            # Check if already cached
            if self.exists(scene_name, idx, 'ldr') and self.exists(scene_name, idx, 'hdr'):
                continue
            
            # Load full image once
            full_ldr = get_image(input_paths[idx])
            full_hdr = LDR2HDR(full_ldr, exposures[idx])
            
            # Create and cache summaries
            sum_ldr = create_global_summary(full_ldr, patch_size)
            sum_hdr = create_global_summary(full_hdr, patch_size)
            
            self.save(scene_name, idx, 'ldr', sum_ldr)
            self.save(scene_name, idx, 'hdr', sum_hdr)


class U2NetDataset(Dataset):
    """
    Optimized Dataset for U2Net HDR reconstruction with disk caching.
    
    Key optimizations:
    - Global summaries cached to disk (computed once per scene)
    - Only patches loaded on-demand during training
    - Minimal memory footprint
    - Fast random access
    """
    def __init__(self, configs,
                 input_name='input_*_aligned.tif',
                 ref_name='ref_*_aligned.tif',
                 input_exp_name='input_exp.txt',
                 ref_exp_name='ref_exp.txt',
                 ref_hdr_name='ref_hdr_aligned.hdr',
                 cache_dir=None,
                 rebuild_cache=False):
        super().__init__()
        print('====> Start preparing U2Net training data.')

        self.filepath = os.path.join(configs.data_path, 'train')
        self.scene_dirs = [scene_dir for scene_dir in os.listdir(self.filepath)
                            if os.path.isdir(os.path.join(self.filepath, scene_dir))]
        self.scene_dirs = sorted(self.scene_dirs)
        self.num_scenes = len(self.scene_dirs)
        self.patch_size = configs.patch_size
        self.patch_stride = configs.patch_stride
        self.num_shots = configs.num_shots
        self.input_name = input_name
        self.ref_name = ref_name
        self.input_exp_name = input_exp_name
        self.ref_exp_name = ref_exp_name
        self.ref_hdr_name = ref_hdr_name
        self.total_count = 0
        
        # Initialize cache
        if cache_dir is None:
            cache_dir = os.path.join(configs.data_path, 'summary_cache')
        self.cache = SummaryCache(cache_dir)
        
        # Store scene metadata and build cache
        self.scene_metadata = []
        self.count = []
        
        for i, scene_dir in enumerate(self.scene_dirs):
            cur_scene_dir = os.path.join(self.filepath, scene_dir)
            in_LDR_paths = sorted(glob(os.path.join(cur_scene_dir, self.input_name)))
            
            # Get image dimensions from first image (only load once)
            tmp_img = get_image(in_LDR_paths[0]).astype(np.float32)
            h, w, c = tmp_img.shape
            
            if h < self.patch_size[0] or w < self.patch_size[1]:
                raise AttributeError('Training images smaller than patch size.')
            
            # Calculate patch count
            h_count = int(np.ceil(h / self.patch_stride))
            w_count = int(np.ceil(w / self.patch_stride))
            patch_count = h_count * w_count
            
            # Load exposures
            in_exp_path = os.path.join(cur_scene_dir, self.input_exp_name)
            in_exp = np.array(open(in_exp_path).read().split('\n')[:self.num_shots]).astype(np.float32)
            in_exp = 2 ** in_exp
            
            # Store metadata
            self.scene_metadata.append({
                'scene_dir': cur_scene_dir,
                'scene_name': scene_dir,
                'input_paths': in_LDR_paths,
                'exposures': in_exp,
                'height': h,
                'width': w,
                'h_count': h_count,
                'w_count': w_count,
                'patch_count': patch_count
            })
            
            self.count.append(patch_count)
            self.total_count += patch_count
            
            # Build cache for this scene if needed
            if rebuild_cache or not self.cache.exists(scene_dir, 0, 'ldr'):
                self.cache.build_scene_cache(
                    cur_scene_dir, scene_dir, in_LDR_paths, 
                    in_exp, self.patch_size, self.num_shots
                )
        
        self.count = np.array(self.count).astype(int)
        self.total_count = int(self.total_count)
        
        print(f'====> Finish preparing U2Net training data!')
        print(f'====> Total patches: {self.total_count}')
        print(f'====> Cache directory: {cache_dir}')

    def __len__(self):
        return self.total_count

    def __getitem__(self, index):
        # Find corresponding scene using cumulative sum for fast lookup
        scene_idx = np.searchsorted(np.cumsum(self.count), index, side='right')
        scene_posidx = index - (np.sum(self.count[:scene_idx]) if scene_idx > 0 else 0)
        
        # Get scene metadata
        meta = self.scene_metadata[scene_idx]
        scene_name = meta['scene_name']
        cur_scene_dir = meta['scene_dir']
        in_LDR_paths = meta['input_paths']
        in_exp = meta['exposures']
        h, w = meta['height'], meta['width']
        h_count, w_count = meta['h_count'], meta['w_count']

        # Calculate patch coordinates
        h_idx = int(scene_posidx // w_count)
        w_idx = int(scene_posidx % w_count)

        h_up = h_idx * self.patch_stride
        h_down = min(h_up + self.patch_size[0], h)
        if h_down == h:
            h_up = h - self.patch_size[0]

        w_left = w_idx * self.patch_stride
        w_right = min(w_left + self.patch_size[1], w)
        if w_right == w:
            w_left = w - self.patch_size[1]

        # ================================================================
        # EXPOSURE PAIR SELECTION
        # ================================================================
        idx1 = self.num_shots // 2  # Middle exposure (motion reference)
        
        if self.num_shots >= 2:
            other_indices = [i for i in range(self.num_shots) if i != idx1]
            idx2 = np.random.choice(other_indices)
        else:
            idx2 = 0

        # ================================================================
        # Load ONLY patches (not full images!)
        # ================================================================
        img1_ldr_patch = get_image(in_LDR_paths[idx1])[h_up:h_down, w_left:w_right, :]
        img2_ldr_patch = get_image(in_LDR_paths[idx2])[h_up:h_down, w_left:w_right, :]
        
        img1_hdr_patch = LDR2HDR(img1_ldr_patch, in_exp[idx1])
        img2_hdr_patch = LDR2HDR(img2_ldr_patch, in_exp[idx2])

        # Load reference HDR patch
        ref_HDR = get_image(os.path.join(cur_scene_dir, self.ref_hdr_name))[h_up:h_down, w_left:w_right, :]

        # ================================================================
        # Load cached summaries (FAST - no full image loading!)
        # ================================================================
        sum1_ldr = self.cache.load(scene_name, idx1, 'ldr')
        sum1_hdr = self.cache.load(scene_name, idx1, 'hdr')
        sum2_ldr = self.cache.load(scene_name, idx2, 'ldr')
        sum2_hdr = self.cache.load(scene_name, idx2, 'hdr')

        # ================================================================
        # Apply random augmentations
        # ================================================================
        distortions = np.random.uniform(0.0, 1.0, 2)
        
        # Horizontal flip
        if distortions[0] < 0.5:
            img1_ldr_patch = np.flip(img1_ldr_patch, axis=1)
            img2_ldr_patch = np.flip(img2_ldr_patch, axis=1)
            img1_hdr_patch = np.flip(img1_hdr_patch, axis=1)
            img2_hdr_patch = np.flip(img2_hdr_patch, axis=1)
            ref_HDR = np.flip(ref_HDR, axis=1)
            sum1_ldr = np.flip(sum1_ldr, axis=1)
            sum1_hdr = np.flip(sum1_hdr, axis=1)
            sum2_ldr = np.flip(sum2_ldr, axis=1)
            sum2_hdr = np.flip(sum2_hdr, axis=1)

        # Rotation
        k = int(distortions[1] * 4 + 0.5)
        img1_ldr_patch = np.rot90(img1_ldr_patch, k)
        img2_ldr_patch = np.rot90(img2_ldr_patch, k)
        img1_hdr_patch = np.rot90(img1_hdr_patch, k)
        img2_hdr_patch = np.rot90(img2_hdr_patch, k)
        ref_HDR = np.rot90(ref_HDR, k)
        sum1_ldr = np.rot90(sum1_ldr, k)
        sum1_hdr = np.rot90(sum1_hdr, k)
        sum2_ldr = np.rot90(sum2_ldr, k)
        sum2_hdr = np.rot90(sum2_hdr, k)
        
        # Concatenate summaries (6 channels each)
        sum1 = np.concatenate([sum1_ldr, sum1_hdr], axis=2)
        sum2 = np.concatenate([sum2_ldr, sum2_hdr], axis=2)
        
        # Transpose to PyTorch format (C, H, W)
        img1_ldr = np.einsum("ijk->kij", img1_ldr_patch)
        img2_ldr = np.einsum("ijk->kij", img2_ldr_patch)
        img1_hdr = np.einsum("ijk->kij", img1_hdr_patch)
        img2_hdr = np.einsum("ijk->kij", img2_hdr_patch)
        sum1 = np.einsum("ijk->kij", sum1)
        sum2 = np.einsum("ijk->kij", sum2)
        ref_HDR = np.einsum("ijk->kij", ref_HDR)

        return (img1_ldr.copy().astype(np.float32), 
                img2_ldr.copy().astype(np.float32),
                img1_hdr.copy().astype(np.float32),
                img2_hdr.copy().astype(np.float32),
                sum1.copy().astype(np.float32), 
                sum2.copy().astype(np.float32),
                ref_HDR.copy().astype(np.float32))


class U2NetTestDataset(Dataset):
    """
    Optimized test dataset with disk caching.
    """
    def __init__(self, configs,
                 input_name='input_*_aligned.tif',
                 input_exp_name='input_exp.txt',
                 ref_hdr_name='ref_hdr_aligned.hdr',
                 cache_dir=None,
                 rebuild_cache=False):
        super().__init__()
        print('====> Start preparing U2Net testing data.')
        
        self.filepath = os.path.join(configs.data_path, 'test')
        self.scene_dirs = [scene_dir for scene_dir in os.listdir(self.filepath)
                            if os.path.isdir(os.path.join(self.filepath, scene_dir))]
        self.scene_dirs = sorted(self.scene_dirs)
        self.num_scenes = len(self.scene_dirs)
        self.num_shots = configs.num_shots
        self.sample_path = configs.sample_dir
        self.input_name = input_name
        self.input_exp_name = input_exp_name
        self.ref_hdr_name = ref_hdr_name
        self.patch_size = configs.patch_size
        
        # Initialize cache
        if cache_dir is None:
            cache_dir = os.path.join(configs.data_path, 'summary_cache_test')
        self.cache = SummaryCache(cache_dir)
        
        # Build cache for test scenes
        for scene_dir in self.scene_dirs:
            scene_path = os.path.join(self.filepath, scene_dir)
            LDR_paths = sorted(glob(os.path.join(scene_path, self.input_name)))
            
            # Load exposures
            exp_path = os.path.join(scene_path, self.input_exp_name)
            ns = len(LDR_paths)
            in_exps = np.array(open(exp_path).read().split('\n')[:ns]).astype(np.float32)
            in_exps = 2 ** in_exps
            
            # Build cache if needed
            if rebuild_cache or not self.cache.exists(scene_dir, 0, 'ldr'):
                self.cache.build_scene_cache(
                    scene_path, scene_dir, LDR_paths,
                    in_exps, self.patch_size, ns
                )
        
        print('====> Finish preparing U2Net testing data!')
        print(f'====> Cache directory: {cache_dir}')

    def __len__(self):
        return self.num_scenes

    def __getitem__(self, index):
        scene_dir = self.scene_dirs[index]
        scene_path = os.path.join(self.filepath, scene_dir)
        sample_path = os.path.join(self.sample_path, scene_dir)
        
        # Load LDR images
        LDR_paths = sorted(glob(os.path.join(scene_path, self.input_name)))
        ns = len(LDR_paths)
        tmp_img = cv2.imread(LDR_paths[0]).astype(np.float32)
        h, w, c = tmp_img.shape
        h = h // 8 * 8
        w = w // 8 * 8

        # Load exposures
        exp_path = os.path.join(scene_path, self.input_exp_name)
        in_exps = np.array(open(exp_path).read().split('\n')[:ns]).astype(np.float32)
        
        # Select exposure indices
        idx1 = ns // 2  # Middle exposure
        idx2 = 0 if idx1 != 0 else 1
        
        # Load the two exposures
        img1_ldr = get_image(LDR_paths[idx1], image_size=[h, w], is_crop=True)
        img2_ldr = get_image(LDR_paths[idx2], image_size=[h, w], is_crop=True)
        
        img1_hdr = LDR2HDR(img1_ldr, 2. ** in_exps[idx1])
        img2_hdr = LDR2HDR(img2_ldr, 2. ** in_exps[idx2])

        # Load reference HDR
        ref_HDR_path = os.path.join(scene_path, self.ref_hdr_name)
        ref_HDR = get_image(ref_HDR_path, image_size=[h, w], is_crop=True)

        # Load cached summaries
        sum1_ldr = self.cache.load(scene_dir, idx1, 'ldr')
        sum1_hdr = self.cache.load(scene_dir, idx1, 'hdr')
        sum2_ldr = self.cache.load(scene_dir, idx2, 'ldr')
        sum2_hdr = self.cache.load(scene_dir, idx2, 'hdr')
        
        # Concatenate summaries
        sum1 = np.concatenate([sum1_ldr, sum1_hdr], axis=2)
        sum2 = np.concatenate([sum2_ldr, sum2_hdr], axis=2)

        # Transpose to PyTorch format
        img1_ldr = np.einsum("ijk->kij", img1_ldr)
        img2_ldr = np.einsum("ijk->kij", img2_ldr)
        img1_hdr = np.einsum("ijk->kij", img1_hdr)
        img2_hdr = np.einsum("ijk->kij", img2_hdr)
        sum1 = np.einsum("ijk->kij", sum1)
        sum2 = np.einsum("ijk->kij", sum2)
        ref_HDR = np.einsum("ijk->kij", ref_HDR)

        return (sample_path,
                img1_ldr.copy().astype(np.float32),
                img2_ldr.copy().astype(np.float32),
                img1_hdr.copy().astype(np.float32),
                img2_hdr.copy().astype(np.float32),
                sum1.copy().astype(np.float32),
                sum2.copy().astype(np.float32),
                ref_HDR.copy().astype(np.float32))