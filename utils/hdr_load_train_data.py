import os
import cv2
import torch
import numpy as np
from glob import glob
from pathlib import Path
from torch.utils.data import Dataset
from utils.tools import get_image, LDR2HDR

# ============================================================
# Memmap helpers
# ============================================================

def save_memmap(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pstr = str(path)
    mm = np.memmap(pstr, dtype=np.float32, mode='w+', shape=array.shape)
    mm[:] = array[:]
    mm.flush()
    del mm


def load_memmap(path, shape):
    # shape must be a tuple of ints
    pstr = str(path)
    return np.memmap(pstr, dtype=np.float32, mode='r', shape=shape)


# ============================================================
# Scene-level cache (HDR images + HDR summaries + GT)
# ============================================================

class SceneCache:
    """
    Disk-backed, memory-efficient cache using np.memmap.
    Stores:
      - Full HDR images   (H, W, 3)  -> {scene}_img{idx}_hdr.dat
      - Global summaries (ph, pw, 3) -> {scene}_sum{idx}_hdr.dat
      - Reference GT HDR  (H, W, 3)  -> {scene}_gt_hdr.dat
      - Metadata:
          - {scene}_meta.npy       -> [H, W, C]
          - {scene}_sum_meta.npy   -> [ph, pw, C]
    """

    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

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

    def exists(self, scene_name, idx):
        return self.img_path(scene_name, idx).exists()

    def gt_exists(self, scene_name):
        return self.gt_path(scene_name).exists()
    
    def clear_scene(self, scene_name):
        for f in self.cache_dir.glob(f"{scene_name}_*"):
            f.unlink()

    def build_scene(self, scene_name, ldr_paths, exposures, patch_size, gt_path):
        print(f"[CACHE] Building HDR memmap cache for scene: {scene_name}")

        summary_shape_saved = False
        hdr_shape_saved = False

        for i, p in enumerate(ldr_paths):
            if self.exists(scene_name, i):
                # if already exists, skip conversion for this exposure
                # but ensure we still have summary/meta saved by some image
                if not summary_shape_saved or not hdr_shape_saved:
                    # try to load shapes from existing files if present
                    if self.meta_path(scene_name).exists():
                        hdr_shape_saved = True
                    if self.sum_meta_path(scene_name).exists():
                        summary_shape_saved = True
                continue

            # Ensure p is a string path
            img = get_image(p).astype(np.float32)  # returns RGB float32
            
            hdr = LDR2HDR(img, exposures[i])        # convert to HDR

            # Save full HDR image as memmap
            save_memmap(self.img_path(scene_name, i), hdr)

            # Build and save summary
            summary = cv2.resize(hdr, patch_size[::-1], interpolation=cv2.INTER_AREA)
            save_memmap(self.sum_path(scene_name, i), summary)

            # Save shapes once
            if not hdr_shape_saved:
                H, W, C = hdr.shape
                np.save(self.meta_path(scene_name), np.array([H, W, C], dtype=np.int32))
                hdr_shape_saved = True

            if not summary_shape_saved:
                np.save(self.sum_meta_path(scene_name), np.array(summary.shape, dtype=np.int32))
                summary_shape_saved = True

        # ---------- GT HDR ----------
        if not self.gt_exists(scene_name):
            # ensure gt_path is string
            gt_arr = get_image(gt_path).astype(np.float32)
            save_memmap(self.gt_path(scene_name), gt_arr)
            # If meta wasn't saved (edge case), save it now
            if not hdr_shape_saved:
                H, W, C = gt_arr.shape
                np.save(self.meta_path(scene_name), np.array([H, W, C], dtype=np.int32))


# ============================================================
# TRAIN DATASET
# ============================================================

class U2NetDataset(Dataset):
    """
    FAST, MEMORY-EFFICIENT U2Net HDR Dataset
    """

    def __init__(self, configs, cache_dir="cache", rebuild_cache=True):
        super().__init__()

        self.root = Path(configs.data_path) / "train"
        self.patch_size = configs.patch_size
        self.stride = configs.patch_stride
        self.num_shots = configs.num_shots

        self.cache = SceneCache(cache_dir)

        self.scenes = []
        self.counts = []

        print("====> Preparing training dataset")

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
                    gt_hdr_path
                )

            # Load shape metadata (must exist after build_scene)
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

        # Memmap images (zero-copy) - full images need exact shape
        img1 = load_memmap(self.cache.img_path(scene_name, idx1), (H, W, 3))
        img2 = load_memmap(self.cache.img_path(scene_name, idx2), (H, W, 3))

        # Load summary shapes from saved meta and use them
        sum_shape = np.load(self.cache.sum_meta_path(scene_name)).astype(int)
        sum_shape = (int(sum_shape[0]), int(sum_shape[1]), int(sum_shape[2]))

        sum1 = load_memmap(self.cache.sum_path(scene_name, idx1), sum_shape)
        sum2 = load_memmap(self.cache.sum_path(scene_name, idx2), sum_shape)

        # Load GT memmap
        gt = load_memmap(self.cache.gt_path(scene_name), (H, W, 3))

        # Patch slicing
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


# ============================================================
# TEST DATASET (same cache, no patches)
# ============================================================

class U2NetTestDataset(Dataset):
    def __init__(self, configs, cache_dir="cache", rebuild_cache=True):
        super().__init__()

        self.root = Path(configs.data_path) / "test"
        self.num_shots = configs.num_shots
        self.patch_size = configs.patch_size
        self.cache = SceneCache(cache_dir)

        self.scenes = []

        for scene_name in sorted(os.listdir(self.root)):
            scene_path = self.root / scene_name
            if not scene_path.is_dir():
                continue

            ldr_paths = sorted(glob(str(scene_path / "input_*_aligned.tif")))
            exp_raw = np.loadtxt(scene_path / "input_exp.txt")[:self.num_shots]
            exposures = 2 ** exp_raw

            gt_hdr_path = scene_path / "ref_hdr_aligned.hdr"

            # Build cache if missing (or forced)
            if rebuild_cache or not self.cache.exists(scene_name, 0):
                self.cache.build_scene(
                    scene_name,
                    ldr_paths,
                    exposures,
                    self.patch_size,
                    gt_hdr_path
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

        # Load full HDR images
        img1 = load_memmap(self.cache.img_path(scene_name, idx1), (H, W, 3))
        img2 = load_memmap(self.cache.img_path(scene_name, idx2), (H, W, 3))

        # Load summaries
        sum_shape = np.load(self.cache.sum_meta_path(scene_name)).astype(int)
        sum_shape = (int(sum_shape[0]), int(sum_shape[1]), int(sum_shape[2]))

        sum1 = load_memmap(self.cache.sum_path(scene_name, idx1), sum_shape)
        sum2 = load_memmap(self.cache.sum_path(scene_name, idx2), sum_shape)

        # 🔹 Load GT HDR
        gt = load_memmap(self.cache.gt_path(scene_name), (H, W, 3))

        # Convert to torch (C,H,W)
        img1 = torch.from_numpy(img1.copy()).permute(2, 0, 1).contiguous()
        img2 = torch.from_numpy(img2.copy()).permute(2, 0, 1).contiguous()
        sum1 = torch.from_numpy(sum1.copy()).permute(2, 0, 1).contiguous()
        sum2 = torch.from_numpy(sum2.copy()).permute(2, 0, 1).contiguous()
        gt   = torch.from_numpy(gt.copy()).permute(2, 0, 1).contiguous()

        return os.path.join('samples/u2net', scene_name), img1, img2, sum1, sum2, gt
