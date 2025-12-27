"""
Complete Fixed Dataset with HDR Summaries
==========================================
This version properly converts summaries to HDR format before caching.

Key changes:
1. Summaries are now HDR (not LDR) to match input format
2. Cache stores HDR summaries
3. All references updated from 'ldr' to 'hdr'
"""

import os
import cv2
from glob import glob
import numpy as np
import pickle
from pathlib import Path
from torch.utils.data import Dataset
from utils.tools import get_image, LDR2HDR


def create_global_summary(full_image, target_size):
    """
    Compress full image into a summary of target_size.
    
    Args:
        full_image: (H, W, C) numpy array (LDR or HDR)
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
    FIXED: Now stores HDR summaries (not LDR)
    """
    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def get_cache_path(self, scene_name, exp_idx, mode='hdr'):
        """Get the cache file path for a specific summary."""
        return self.cache_dir / f"{scene_name}_exp{exp_idx}_{mode}.npy"
    
    def exists(self, scene_name, exp_idx, mode='hdr'):
        """Check if a cached summary exists."""
        return self.get_cache_path(scene_name, exp_idx, mode).exists()
    
    def save(self, scene_name, exp_idx, mode, data):
        """Save a summary to disk."""
        cache_path = self.get_cache_path(scene_name, exp_idx, mode)
        np.save(cache_path, data.astype(np.float32))
    
    def load(self, scene_name, exp_idx, mode='hdr'):
        """Load a summary from disk."""
        cache_path = self.get_cache_path(scene_name, exp_idx, mode)
        return np.load(cache_path)
    
    def build_scene_cache(self, scene_path, scene_name, input_paths, exposures, 
                          patch_size, num_shots):
        """
        FIXED: Build HDR summaries instead of LDR summaries.
        
        This ensures summaries match the format of input patches used in training.
        """
        print(f"Building HDR cache for scene: {scene_name}")
        
        for idx in range(num_shots):
            # Check if already cached
            if self.exists(scene_name, idx, 'hdr'):
                continue
            
            # Load full LDR image
            full_ldr = get_image(input_paths[idx])
            
            # CRITICAL FIX: Convert to HDR with exposure BEFORE downsampling
            full_hdr = LDR2HDR(full_ldr, exposures[idx])
            
            # Downsample the HDR version
            sum_hdr = create_global_summary(full_hdr, patch_size)
            
            # Save as HDR
            self.save(scene_name, idx, 'hdr', sum_hdr)


class U2NetDataset(Dataset):
    """
    Optimized Dataset for U2Net HDR reconstruction with disk caching.
    
    ALL FIXES APPLIED:
    - Summaries now in HDR format (matches input patches)
    - Ensures idx1 != idx2 for different exposures
    - Validates exposure differences
    - Proper log compression for HDR images
    """
    def __init__(self, configs,
                 input_name='input_*_aligned.tif',
                 ref_name='ref_*_aligned.tif',
                 input_exp_name='input_exp.txt',
                 ref_exp_name='ref_exp.txt',
                 ref_hdr_name='ref_hdr_aligned.hdr',
                 cache_dir=None,
                 rebuild_cache=False,
                 debug=False):
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
        self.debug = debug
        
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
            in_exp_raw = np.array(open(in_exp_path).read().split('\n')[:self.num_shots]).astype(np.float32)
            in_exp = 2 ** in_exp_raw  # Convert to multipliers
            
            # VALIDATION: Check exposure diversity
            if i == 0:  # Print for first scene
                print(f"\n  First scene exposure check:")
                print(f"    Raw values: {in_exp_raw}")
                print(f"    Multipliers (2^exp): {in_exp}")
                print(f"    Ratio (max/min): {in_exp.max() / in_exp.min():.4f}")
                
                if len(np.unique(in_exp_raw)) == 1:
                    print(f"    ❌ WARNING: All exposures are IDENTICAL in {scene_dir}!")
                elif in_exp.max() / in_exp.min() < 1.5:
                    print(f"    ⚠️  WARNING: Small exposure variation in {scene_dir}")
            
            # Store metadata
            self.scene_metadata.append({
                'scene_dir': cur_scene_dir,
                'scene_name': scene_dir,
                'input_paths': in_LDR_paths,
                'exposures': in_exp,
                'exposure_raw': in_exp_raw,
                'height': h,
                'width': w,
                'h_count': h_count,
                'w_count': w_count,
                'patch_count': patch_count
            })
            
            self.count.append(patch_count)
            self.total_count += patch_count
            
            # FIXED: Check for HDR cache (not LDR)
            if rebuild_cache or not self.cache.exists(scene_dir, 0, 'hdr'):
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
        in_exp_raw = meta['exposure_raw']
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

        # Select two DIFFERENT exposures
        idx1 = self.num_shots // 2  # Middle exposure (motion reference)
        
        if self.num_shots >= 2:
            other_indices = [i for i in range(self.num_shots) if i != idx1]
            idx2 = np.random.choice(other_indices)
        else:
            idx2 = 0
            print("⚠️  WARNING: Only 1 exposure available, idx1==idx2!")
        
        # DEBUG: First time loading, print exposure info
        if self.debug and index == 0:
            print(f"\n=== FIRST BATCH DEBUG INFO ===")
            print(f"Scene: {scene_name}")
            print(f"Selected indices: idx1={idx1}, idx2={idx2}")
            print(f"Raw exposures: exp1={in_exp_raw[idx1]:.4f}, exp2={in_exp_raw[idx2]:.4f}")
            print(f"Exposure multipliers: exp1={in_exp[idx1]:.4f}, exp2={in_exp[idx2]:.4f}")
            print(f"Ratio (exp1/exp2): {in_exp[idx1] / in_exp[idx2]:.4f}")
            print(f"==============================\n")

        # Load LDR patches
        img1_ldr = get_image(in_LDR_paths[idx1])[h_up:h_down, w_left:w_right, :]
        img2_ldr = get_image(in_LDR_paths[idx2])[h_up:h_down, w_left:w_right, :]
        
        # Load reference HDR patch (will be auto-converted with log compression)
        ref_HDR = get_image(os.path.join(cur_scene_dir, self.ref_hdr_name))[h_up:h_down, w_left:w_right, :]
        
        # FIXED: Load HDR summaries (not LDR)
        sum1 = self.cache.load(scene_name, idx1, 'hdr')
        sum2 = self.cache.load(scene_name, idx2, 'hdr')
        
        # Convert LDR patches to HDR with DIFFERENT exposures
        img1_hdr = LDR2HDR(img1_ldr, in_exp[idx1])
        img2_hdr = LDR2HDR(img2_ldr, in_exp[idx2])
        
        # DEBUG: Check if conversion worked
        if self.debug and index == 0:
            print(f"After LDR2HDR conversion:")
            print(f"  img1_hdr range: [{img1_hdr.min():.4f}, {img1_hdr.max():.4f}]")
            print(f"  img2_hdr range: [{img2_hdr.min():.4f}, {img2_hdr.max():.4f}]")
            print(f"  sum1 range: [{sum1.min():.4f}, {sum1.max():.4f}]")
            print(f"  sum2 range: [{sum2.min():.4f}, {sum2.max():.4f}]")
            print(f"  ref_HDR range: [{ref_HDR.min():.4f}, {ref_HDR.max():.4f}]")
            print(f"  ✓ All in HDR format now!")
        
        # Data augmentation
        distortions = np.random.uniform(0.0, 1.0, 2)
        
        # Horizontal flip
        if distortions[0] < 0.5:
            img1_hdr = np.flip(img1_hdr, axis=1)
            img2_hdr = np.flip(img2_hdr, axis=1)
            sum1 = np.flip(sum1, axis=1)
            sum2 = np.flip(sum2, axis=1)
            ref_HDR = np.flip(ref_HDR, axis=1)

        # Rotation
        k = int(distortions[1] * 4 + 0.5)
        img1_hdr = np.rot90(img1_hdr, k)
        img2_hdr = np.rot90(img2_hdr, k)
        sum1 = np.rot90(sum1, k)
        sum2 = np.rot90(sum2, k)
        ref_HDR = np.rot90(ref_HDR, k)

        # Transpose to PyTorch format (C, H, W)
        img1_hdr = np.einsum("ijk->kij", img1_hdr)
        img2_hdr = np.einsum("ijk->kij", img2_hdr)
        sum1 = np.einsum("ijk->kij", sum1)
        sum2 = np.einsum("ijk->kij", sum2)
        ref_HDR = np.einsum("ijk->kij", ref_HDR)

        return (img1_hdr.copy().astype(np.float32), 
                img2_hdr.copy().astype(np.float32),
                sum1.copy().astype(np.float32),
                sum2.copy().astype(np.float32),
                ref_HDR.copy().astype(np.float32))


class U2NetTestDataset(Dataset):
    """
    Optimized test dataset with disk caching.
    FIXED: Now uses HDR summaries
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
            in_exps_raw = np.array(open(exp_path).read().split('\n')[:ns]).astype(np.float32)
            in_exps = 2 ** in_exps_raw
            
            # FIXED: Check for HDR cache
            if rebuild_cache or not self.cache.exists(scene_dir, 0, 'hdr'):
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
        in_exps_raw = np.array(open(exp_path).read().split('\n')[:ns]).astype(np.float32)
        in_exps = 2 ** in_exps_raw
        
        # Select exposure indices (ensure they're different)
        idx1 = ns // 2  # Middle exposure
        idx2 = 0 if idx1 != 0 else 1  # Ensure idx2 != idx1
        
        # Load the two exposures
        img1_ldr = get_image(LDR_paths[idx1], image_size=[h, w], is_crop=True)
        img2_ldr = get_image(LDR_paths[idx2], image_size=[h, w], is_crop=True)
 
        # Convert to HDR with correct exposures
        img1_hdr = LDR2HDR(img1_ldr, in_exps[idx1])
        img2_hdr = LDR2HDR(img2_ldr, in_exps[idx2])
        
        # Load reference HDR
        ref_HDR_path = os.path.join(scene_path, self.ref_hdr_name)
        ref_HDR = get_image(ref_HDR_path, image_size=[h, w], is_crop=True)

        # FIXED: Load HDR summaries
        sum1 = self.cache.load(scene_dir, idx1, 'hdr')
        sum2 = self.cache.load(scene_dir, idx2, 'hdr')
        
        # Transpose to PyTorch format
        img1_hdr = np.einsum("ijk->kij", img1_hdr)
        img2_hdr = np.einsum("ijk->kij", img2_hdr)
        sum1 = np.einsum("ijk->kij", sum1)
        sum2 = np.einsum("ijk->kij", sum2)
        ref_HDR = np.einsum("ijk->kij", ref_HDR)

        return (sample_path,
                img1_hdr.copy().astype(np.float32),
                img2_hdr.copy().astype(np.float32),
                sum1.copy().astype(np.float32),
                sum2.copy().astype(np.float32),
                ref_HDR.copy().astype(np.float32))