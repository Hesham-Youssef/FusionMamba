import os
import cv2
import torch
import numpy as np
from glob import glob
from pathlib import Path
from torch.utils.data import Dataset

def generate_shuffle_pattern(H, W, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    total_pixels = H * W
    shuffle_indices = np.random.permutation(total_pixels)
    
    # Create inverse permutation for unshuffling
    unshuffle_indices = np.empty_like(shuffle_indices)
    unshuffle_indices[shuffle_indices] = np.arange(total_pixels)
    
    return shuffle_indices, unshuffle_indices


def shuffle_image(img, shuffle_indices, H, W):
    C = img.shape[2]
    img_flat = img.reshape(-1, C)  # (H*W, C)
    img_shuffled = img_flat[shuffle_indices]  # Apply permutation
    return img_shuffled.reshape(H, W, C)


def unshuffle_image(img, unshuffle_indices, H, W):
    is_torch = isinstance(img, torch.Tensor)
    
    if is_torch:
        # Handle torch tensor (C, H, W)
        device = img.device
        C = img.shape[0]
        unshuffle_indices_torch = torch.from_numpy(unshuffle_indices).to(device)
        
        # Reshape and apply unshuffle to all channels at once
        img_flat = img.reshape(C, -1)  # (C, H*W)
        img_unshuffled = img_flat[:, unshuffle_indices_torch]  # (C, H*W)
        return img_unshuffled.reshape(C, H, W)
    else:
        # Handle numpy array (H, W, C)
        C = img.shape[2]
        img_flat = img.reshape(-1, C)  # (H*W, C)
        img_unshuffled = img_flat[unshuffle_indices]
        return img_unshuffled.reshape(H, W, C)


def save_memmap(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pstr = str(path)
    mm = np.memmap(pstr, dtype=np.float32, mode='w+', shape=array.shape)
    mm[:] = array[:]
    mm.flush()
    del mm


def load_memmap(path, shape):
    pstr = str(path)
    return np.memmap(pstr, dtype=np.float32, mode='r', shape=shape)


class SceneCache:
    def __init__(self, cache_dir, enable_shuffle=True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_shuffle = enable_shuffle

    def img_path(self, scene_name, idx):
        return self.cache_dir / f"{scene_name}_img{idx}_hdr.dat"

    def sum_path(self, scene_name, idx):
        return self.cache_dir / f"{scene_name}_sum{idx}_hdr.dat"

    def meta_path(self, scene_name):
        return self.cache_dir / f"{scene_name}_meta.npy"

    def sum_meta_path(self, scene_name):
        return self.cache_dir / f"{scene_name}_sum_meta.npy"

    def gt_path(self, scene_name):
        return self.cache_dir / f"{scene_name}_gt_hdr.dat"
    
    def shuffle_path(self, scene_name):
        """Path to shuffle pattern"""
        return self.cache_dir / f"{scene_name}_shuffle.npy"
    
    def unshuffle_path(self, scene_name):
        """Path to unshuffle (inverse) pattern"""
        return self.cache_dir / f"{scene_name}_unshuffle.npy"

    def exists(self, scene_name, idx):
        return self.img_path(scene_name, idx).exists()

    def gt_exists(self, scene_name):
        return self.gt_path(scene_name).exists()
    
    def shuffle_exists(self, scene_name):
        return self.shuffle_path(scene_name).exists()
    
    def clear_scene(self, scene_name):
        """Clear all cached files for a scene"""
        for f in self.cache_dir.glob(f"{scene_name}_*"):
            f.unlink()

    def save_shuffle_pattern(self, scene_name, shuffle_indices, unshuffle_indices):
        """Save shuffle and unshuffle patterns"""
        np.save(self.shuffle_path(scene_name), shuffle_indices)
        np.save(self.unshuffle_path(scene_name), unshuffle_indices)
    
    def load_shuffle_pattern(self, scene_name):
        """Load shuffle pattern (returns shuffle and unshuffle indices)"""
        shuffle_indices = np.load(self.shuffle_path(scene_name))
        unshuffle_indices = np.load(self.unshuffle_path(scene_name))
        return shuffle_indices, unshuffle_indices

    def build_scene(self, scene_name, ldr_paths, exposures, patch_size, gt_path, 
               get_image_fn, ldr2hdr_fn, shuffle_seed=None):
        print(f"[CACHE] Building HDR memmap cache for scene: {scene_name}")
        if self.enable_shuffle:
            print(f"[CACHE] Pixel shuffling ENABLED for scene: {scene_name}")

        summary_shape_saved = False
        hdr_shape_saved = False
        shuffle_pattern_created = False

        for i, p in enumerate(ldr_paths):
            if self.exists(scene_name, i):
                if not summary_shape_saved or not hdr_shape_saved:
                    if self.meta_path(scene_name).exists():
                        hdr_shape_saved = True
                    if self.sum_meta_path(scene_name).exists():
                        summary_shape_saved = True
                if self.enable_shuffle and not shuffle_pattern_created:
                    if self.shuffle_exists(scene_name):
                        shuffle_pattern_created = True
                continue

            # Load and convert image
            img = get_image_fn(p).astype(np.float32)
            hdr = ldr2hdr_fn(img, exposures[i])
            H, W, C = hdr.shape

            # ============================================================
            # ✅ CRITICAL FIX: Create summary BEFORE shuffling!
            # ============================================================
            # The global summary needs spatial structure to preserve
            # the full range including negative values
            summary = cv2.resize(hdr, patch_size[::-1], interpolation=cv2.INTER_AREA)
            save_memmap(self.sum_path(scene_name, i), summary)
            
            # Save summary shape once
            if not summary_shape_saved:
                np.save(self.sum_meta_path(scene_name), np.array(summary.shape, dtype=np.int32))
                summary_shape_saved = True
            
            print(f"[CACHE] Created global summary: range=[{summary.min():.3f}, {summary.max():.3f}]")

            # Generate shuffle pattern once per scene
            if self.enable_shuffle and not shuffle_pattern_created:
                # Use scene_name hash as seed if no seed provided for consistency
                if shuffle_seed is None:
                    shuffle_seed = hash(scene_name) % (2**32)
                
                shuffle_indices, unshuffle_indices = generate_shuffle_pattern(H, W, shuffle_seed)
                self.save_shuffle_pattern(scene_name, shuffle_indices, unshuffle_indices)
                shuffle_pattern_created = True
                print(f"[CACHE] Generated shuffle pattern with seed: {shuffle_seed}")
            
            # ============================================================
            # ✅ NOW shuffle the main image (AFTER creating summary)
            # ============================================================
            if self.enable_shuffle:
                if not shuffle_pattern_created:
                    # Load existing pattern
                    shuffle_indices, unshuffle_indices = self.load_shuffle_pattern(scene_name)
                hdr = shuffle_image(hdr, shuffle_indices, H, W)
                print(f"[CACHE] Shuffled HDR image")

            # Save shuffled HDR image as memmap
            save_memmap(self.img_path(scene_name, i), hdr)

            # Save HDR shape once
            if not hdr_shape_saved:
                np.save(self.meta_path(scene_name), np.array([H, W, C], dtype=np.int32))
                hdr_shape_saved = True

        # ---------- GT HDR ----------
        if not self.gt_exists(scene_name):
            gt_arr = get_image_fn(gt_path).astype(np.float32)
            H, W, C = gt_arr.shape
            
            # ✅ Apply SAME shuffle pattern to GT (but GT doesn't need summary)
            if self.enable_shuffle:
                if not shuffle_pattern_created:
                    shuffle_indices, unshuffle_indices = self.load_shuffle_pattern(scene_name)
                gt_arr = shuffle_image(gt_arr, shuffle_indices, H, W)
            
            save_memmap(self.gt_path(scene_name), gt_arr)
            
            if not hdr_shape_saved:
                np.save(self.meta_path(scene_name), np.array([H, W, C], dtype=np.int32))


class U2NetDataset(Dataset):
    def __init__(self, configs, cache_dir="cache", rebuild_cache=False, 
                 enable_shuffle=True, get_image_fn=None, ldr2hdr_fn=None):
        super().__init__()

        self.root = Path(configs.data_path) / "train"
        self.patch_size = configs.patch_size
        self.stride = configs.patch_stride
        self.num_shots = configs.num_shots
        self.enable_shuffle = enable_shuffle

        self.cache = SceneCache(cache_dir, enable_shuffle=enable_shuffle)

        # Import functions if not provided
        if get_image_fn is None:
            from utils.tools import get_image
            get_image_fn = get_image
        if ldr2hdr_fn is None:
            from utils.tools import LDR2HDR
            ldr2hdr_fn = LDR2HDR

        self.scenes = []
        self.counts = []

        print("====> Preparing training dataset")
        if enable_shuffle:
            print("====> Pixel shuffling ENABLED")

        for scene_name in sorted(os.listdir(self.root)):
            scene_path = self.root / scene_name
            if not scene_path.is_dir():
                continue

            ldr_paths = sorted(glob(str(scene_path / "input_*_aligned.tif")))
            exp_raw = np.loadtxt(scene_path / "input_exp.txt")[:self.num_shots]
            exposures = 2 ** exp_raw

            gt_hdr_path = scene_path / "ref_hdr_aligned.hdr"

            if rebuild_cache:
                self.cache.clear_scene(scene_name)

            if rebuild_cache or not self.cache.exists(scene_name, 0):
                self.cache.build_scene(
                    scene_name,
                    ldr_paths,
                    exposures,
                    self.patch_size,
                    gt_hdr_path,
                    get_image_fn,
                    ldr2hdr_fn
                )

            # Load shape metadata
            HWC = np.load(self.cache.meta_path(scene_name)).astype(int)
            H, W, C = int(HWC[0]), int(HWC[1]), int(HWC[2])

            h_cnt = int(np.ceil(H / self.stride))
            w_cnt = int(np.ceil(W / self.stride))
            patch_count = h_cnt * w_cnt

            self.scenes.append({
                "name": scene_name,
                "H": H,
                "W": W,
                "h_cnt": h_cnt,
                "w_cnt": w_cnt
            })
            self.counts.append(patch_count)

        self.counts = np.array(self.counts, dtype=np.int64)
        self.total = int(self.counts.sum())

        print(f"====> Total patches: {self.total}")
        print(f"====> Cache dir: {cache_dir}")

    def __len__(self):
        return self.total

    def __getitem__(self, index):
        # Locate scene
        scene_idx = np.searchsorted(np.cumsum(self.counts), index, side='right')
        local_idx = index - (self.counts[:scene_idx].sum() if scene_idx > 0 else 0)

        scene = self.scenes[scene_idx]
        scene_name = scene["name"]
        H, W = scene["H"], scene["W"]

        h_idx = local_idx // scene["w_cnt"]
        w_idx = local_idx % scene["w_cnt"]

        h0 = min(h_idx * self.stride, H - self.patch_size[0])
        w0 = min(w_idx * self.stride, W - self.patch_size[1])

        # Exposure selection
        idx1 = self.num_shots // 2
        idx2 = np.random.choice([i for i in range(self.num_shots) if i != idx1])

        # Load memmap images (already shuffled if enabled)
        img1 = load_memmap(self.cache.img_path(scene_name, idx1), (H, W, 3))
        img2 = load_memmap(self.cache.img_path(scene_name, idx2), (H, W, 3))

        sum_shape = np.load(self.cache.sum_meta_path(scene_name)).astype(int)
        sum_shape = (int(sum_shape[0]), int(sum_shape[1]), int(sum_shape[2]))

        sum1 = load_memmap(self.cache.sum_path(scene_name, idx1), sum_shape)
        sum2 = load_memmap(self.cache.sum_path(scene_name, idx2), sum_shape)

        gt = load_memmap(self.cache.gt_path(scene_name), (H, W, 3))

        # Patch slicing (from shuffled images)
        p1 = img1[h0:h0+self.patch_size[0], w0:w0+self.patch_size[1]]
        p2 = img2[h0:h0+self.patch_size[0], w0:w0+self.patch_size[1]]
        gt_patch = gt[h0:h0+self.patch_size[0], w0:w0+self.patch_size[1]]

        # Convert to torch (C,H,W)
        p1 = torch.from_numpy(p1.copy()).permute(2, 0, 1).contiguous()
        p2 = torch.from_numpy(p2.copy()).permute(2, 0, 1).contiguous()
        s1 = torch.from_numpy(sum1.copy()).permute(2, 0, 1).contiguous()
        s2 = torch.from_numpy(sum2.copy()).permute(2, 0, 1).contiguous()
        gt_patch = torch.from_numpy(gt_patch.copy()).permute(2, 0, 1).contiguous()

        return p1, p2, s1, s2, gt_patch


class U2NetTestDataset(Dataset):
    def __init__(self, configs, cache_dir="cache", rebuild_cache=False, 
                 enable_shuffle=True, get_image_fn=None, ldr2hdr_fn=None):
        super().__init__()

        self.root = Path(configs.data_path) / "test"
        self.num_shots = configs.num_shots
        self.patch_size = configs.patch_size
        self.enable_shuffle = enable_shuffle
        
        self.cache = SceneCache(cache_dir, enable_shuffle=enable_shuffle)

        # Import functions if not provided
        if get_image_fn is None:
            from utils.tools import get_image
            get_image_fn = get_image
        if ldr2hdr_fn is None:
            from utils.tools import LDR2HDR
            ldr2hdr_fn = LDR2HDR

        self.scenes = []

        for scene_name in sorted(os.listdir(self.root)):
            scene_path = self.root / scene_name
            if not scene_path.is_dir():
                continue

            ldr_paths = sorted(glob(str(scene_path / "input_*_aligned.tif")))
            exp_raw = np.loadtxt(scene_path / "input_exp.txt")[:self.num_shots]
            exposures = 2 ** exp_raw

            gt_hdr_path = scene_path / "ref_hdr_aligned.hdr"

            if rebuild_cache or not self.cache.exists(scene_name, 0):
                self.cache.build_scene(
                    scene_name,
                    ldr_paths,
                    exposures,
                    self.patch_size,
                    gt_hdr_path,
                    get_image_fn,
                    ldr2hdr_fn
                )

            self.scenes.append(scene_name)

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene_name = self.scenes[idx]

        # Load image shape
        HWC = np.load(self.cache.meta_path(scene_name)).astype(int)
        H, W, C = int(HWC[0]), int(HWC[1]), int(HWC[2])

        # Exposure selection
        idx1 = self.num_shots // 2
        idx2 = 0 if idx1 != 0 else 1

        # Load full HDR images (shuffled)
        img1 = load_memmap(self.cache.img_path(scene_name, idx1), (H, W, 3))
        img2 = load_memmap(self.cache.img_path(scene_name, idx2), (H, W, 3))

        sum_shape = np.load(self.cache.sum_meta_path(scene_name)).astype(int)
        sum_shape = (int(sum_shape[0]), int(sum_shape[1]), int(sum_shape[2]))

        sum1 = load_memmap(self.cache.sum_path(scene_name, idx1), sum_shape)
        sum2 = load_memmap(self.cache.sum_path(scene_name, idx2), sum_shape)

        gt = load_memmap(self.cache.gt_path(scene_name), (H, W, 3))

        # Convert to torch (C,H,W)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).contiguous()
        img2 = torch.from_numpy(img2.copy()).permute(2, 0, 1).contiguous()
        sum1 = torch.from_numpy(sum1.copy()).permute(2, 0, 1).contiguous()
        sum2 = torch.from_numpy(sum2.copy()).permute(2, 0, 1).contiguous()
        gt = torch.from_numpy(gt.copy()).permute(2, 0, 1).contiguous()

        # Load unshuffle pattern if shuffling was enabled
        # FIX: Always return a tensor instead of None to avoid collate errors
        if self.enable_shuffle and self.cache.shuffle_exists(scene_name):
            _, unshuffle_indices = self.cache.load_shuffle_pattern(scene_name)
            unshuffle_indices = torch.from_numpy(unshuffle_indices).long()
        else:
            # Return empty tensor when shuffling is disabled
            unshuffle_indices = torch.empty(0, dtype=torch.long)

        # Return dimensions as tensors for proper batching
        H_tensor = torch.tensor(H, dtype=torch.long)
        W_tensor = torch.tensor(W, dtype=torch.long)

        return os.path.join('samples/u2net', scene_name), img1, img2, sum1, sum2, gt, unshuffle_indices, (H_tensor, W_tensor)

# ============================================================
# Utility function for unshuffling outputs
# ============================================================

def unshuffle_output(output, unshuffle_indices, H, W):
    if unshuffle_indices is None:
        return output
    
    B, C = output.shape[:2]
    device = output.device
    
    # Convert indices to torch once
    unshuffle_indices_torch = torch.from_numpy(unshuffle_indices).to(device)
    
    unshuffled = torch.zeros_like(output)
    
    for b in range(B):
        # Reshape (C, H, W) → (C, H*W) to apply pixel-wise permutation
        img_flat = output[b].reshape(C, -1)  # (C, H*W)
        
        # Apply unshuffle to all channels at once
        # This moves entire pixels (all channels) together
        img_unshuffled = img_flat[:, unshuffle_indices_torch]  # (C, H*W)
        
        # Reshape back to (C, H, W)
        unshuffled[b] = img_unshuffled.reshape(C, H, W)
    
    return unshuffled