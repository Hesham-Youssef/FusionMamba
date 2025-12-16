# hdr_load_train_data.py
# Maximum optimization with parallel loading, shared memory, half-precision, and disk caching

import os
import glob
import math
import csv
import pickle
from pathlib import Path
from functools import lru_cache
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torch.utils.data as data


# -----------------------------
# Configuration
# -----------------------------
class Config:
    USE_HALF_PRECISION = False  # Store in fp16 to save 50% memory
    NUM_LOAD_THREADS = 4  # Parallel image loading
    PREFETCH_SIZE = 2  # Number of pairs to prefetch
    USE_DISK_CACHE = False  # Cache preprocessed tiles to disk
    DISK_CACHE_DIR = None


# -----------------------------
# Utils
# -----------------------------

def _to_tensor_fast(image_np, is_hdr=False, use_half=False):
    """Optimized tensor conversion with optional fp16."""
    # Avoid extra copy by checking dtype first
    if image_np.dtype != np.float32:
        img = image_np.astype(np.float32)
    else:
        img = image_np
    
    if not is_hdr and img.max() > 1.5:
        img = img * (1.0 / 255.0)  # Slightly faster than division
    
    if img.ndim == 2:
        img = img[:, :, None]
    
    # Direct memory view when possible
    tensor = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1)))
    
    if use_half:
        return tensor.half()
    return tensor.float()


@lru_cache(maxsize=64)
def make_hann_window(h, w, device_str='cpu', use_half=False):
    """Cached Hann window with fp16 support."""
    device = torch.device(device_str)
    dtype = torch.float16 if use_half else torch.float32
    
    wh = torch.hann_window(h, periodic=False, dtype=dtype, device=device) if h > 1 else torch.ones(1, dtype=dtype, device=device)
    ww = torch.hann_window(w, periodic=False, dtype=dtype, device=device) if w > 1 else torch.ones(1, dtype=dtype, device=device)
    return wh.unsqueeze(1) @ ww.unsqueeze(0)


def load_image_fast(path):
    """Fast image loading with BGR to RGB."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read {path}")
    # Vectorized color conversion
    if img.ndim == 3 and img.shape[2] == 3:
        return img[:, :, ::-1].copy()  # BGR to RGB via slicing
    return img


# -----------------------------
# Parallel Image Loader
# -----------------------------
class ParallelImageLoader:
    """Thread pool for parallel image loading."""
    
    def __init__(self, num_threads=4):
        self.executor = ThreadPoolExecutor(max_workers=num_threads)
        self._cache = {}
        self._lock = threading.Lock()
    
    def load_triplet(self, ldr1_path, ldr2_path, hdr_path):
        """Load three images in parallel."""
        futures = [
            self.executor.submit(load_image_fast, ldr1_path),
            self.executor.submit(load_image_fast, ldr2_path),
            self.executor.submit(load_image_fast, hdr_path)
        ]
        return [f.result() for f in futures]
    
    def shutdown(self):
        self.executor.shutdown(wait=True)


class DiskCacheManager:
    """
    Disk cache manager with safe cross-process writes and size enforcement.

    Key features:
      - Writes a unique tmp file per writer (pid + uuid) to avoid tmp-name races.
      - Uses a global lockfile (INDEX_NAME + '.lock') acquired via O_EXCL to
        serialize index updates and finalization (tmp -> final).
      - While holding the lock, evicts old files as needed to keep total size
        <= max_size_bytes before promoting the tmp file into the cache.
      - summary_only: store only sum1, sum2 and a tiny preview (keeps cache small).
    """

    INDEX_NAME = "cache_index.json"

    def __init__(self, cache_dir, max_size_bytes=None,
                 summary_only=False, downsample_factor=8, use_half=False):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_size_bytes = int(max_size_bytes) if max_size_bytes is not None else None
        self.summary_only = bool(summary_only)
        self.downsample_factor = max(1, int(downsample_factor))
        self.use_half = bool(use_half)

        self._lock = threading.Lock()         # in-process lock for index ops
        self._index = {}
        if self.cache_dir:
            # create directory (race-safe)
            try:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            self._index_path = self.cache_dir / self.INDEX_NAME
            self._load_index()

    # ---------- internal helpers ----------
    def _load_index(self):
        try:
            if self.cache_dir and self._index_path.exists():
                with open(self._index_path, 'r') as f:
                    self._index = json.load(f)
            else:
                self._index = {}
        except Exception:
            self._index = {}

    def _save_index(self):
        try:
            if not self.cache_dir:
                return
            with open(self._index_path, 'w') as f:
                json.dump(self._index, f)
        except Exception:
            pass

    def _make_filename(self, pair_idx, tile_params):
        h, w, sh, sw = tile_params
        mode = "sum" if self.summary_only else "full"
        return f"pair_{pair_idx}_t{h}x{w}_s{sh}x{sw}_{mode}.npz"

    def _get_total_size(self):
        total = 0
        for v in self._index.values():
            total += int(v.get("size", 0))
        return total

    def _evict_until_fit(self, required_bytes):
        """
        Evict oldest files until there's room for required_bytes (or until index empty).
        Must be called while holding the global lock (see save()).
        """
        if self.max_size_bytes is None:
            return

        total = self._get_total_size()
        target = int(self.max_size_bytes)
        if total + required_bytes <= target:
            return

        # sort by atime (oldest first)
        items = sorted(self._index.items(), key=lambda kv: kv[1].get("atime", 0))
        for name, meta in items:
            if total + required_bytes <= target:
                break
            try:
                p = self.cache_dir / name
                if p.exists():
                    p.unlink()
                total -= int(meta.get("size", 0))
            except Exception:
                # continue even if removal fails
                total -= int(meta.get("size", 0))
            self._index.pop(name, None)

        self._save_index()

    # ---------- public API ----------
    def load(self, pair_idx, tile_params):
        """
        Load cached data (returns dict of torch tensors or None).
        If summary_only was used when saving, returned dict may only contain
        sum1/sum2/preview keys.
        """
        if not self.cache_dir:
            return None

        fname = self._make_filename(pair_idx, tile_params)
        p = self.cache_dir / fname
        if not p.exists():
            return None

        try:
            npz = np.load(str(p), allow_pickle=False)
            out = {}
            for k in npz.files:
                arr = npz[k]
                out[k] = torch.from_numpy(arr)
            # update access time in index (best-effort)
            with self._lock:
                meta = self._index.get(fname, {})
                meta["atime"] = time.time()
                self._index[fname] = meta
                self._save_index()
            return out
        except Exception:
            return None

    def save(self, pair_idx, tile_params, data: dict, lock_timeout=1.5):
        """
        Save `data` into cache (safe cross-process).
        Steps:
          1. prepare small `to_save` dict (numpy arrays).
          2. write unique tmp file (pid+uuid).
          3. attempt to acquire global lockfile (INDEX_NAME + '.lock') within lock_timeout.
             - If lock acquired: evict old files (if needed), move tmp -> final (os.replace),
               update index, release lock.
             - If lock not acquired within timeout: remove tmp and skip (fast path).
        """
        if not self.cache_dir:
            return

        import os
        import time
        from uuid import uuid4

        fname = self._make_filename(pair_idx, tile_params)
        p = self.cache_dir / fname
        lock_path = self.cache_dir / (self.INDEX_NAME + ".lock")

        # --- build to_save dict (small, summary-only preferred) ---
        to_save = {}
        try:
            if self.summary_only:
                # sums
                if 'sum1' in data:
                    try:
                        a = data['sum1'].cpu().numpy()
                        if self.use_half:
                            a = a.astype(np.float16)
                        to_save['sum1'] = a
                    except Exception:
                        pass
                if 'sum2' in data:
                    try:
                        a = data['sum2'].cpu().numpy()
                        if self.use_half:
                            a = a.astype(np.float16)
                        to_save['sum2'] = a
                    except Exception:
                        pass
                # tiny preview (first tile heavily downsampled) - optional
                if 'tiles_g' in data:
                    try:
                        tiles_g = data['tiles_g']
                        if hasattr(tiles_g, "shape") and tiles_g.shape[0] > 0:
                            tiles_g0 = tiles_g[0:1].cpu()
                            preview = F.adaptive_avg_pool2d(
                                tiles_g0,
                                (max(1, tiles_g0.shape[-2] // self.downsample_factor),
                                 max(1, tiles_g0.shape[-1] // self.downsample_factor))
                            )
                            arr = preview.numpy()
                            if self.use_half:
                                arr = arr.astype(np.float16)
                            to_save['preview'] = arr
                    except Exception:
                        pass
            else:
                # full cache (rare)
                for k, v in data.items():
                    try:
                        if isinstance(v, torch.Tensor):
                            a = v.cpu().numpy()
                        else:
                            a = np.asarray(v)
                        if self.use_half and a.dtype == np.float32:
                            a = a.astype(np.float16)
                        to_save[k] = a
                    except Exception:
                        continue
        except Exception:
            # if anything goes wrong preparing arrays, abort save
            return

        # ensure cache dir exists
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # write unique tmp file
        tmp = self.cache_dir / (p.name + f".tmp.{os.getpid()}.{uuid4().hex}")
        try:
            np.savez_compressed(str(tmp), **to_save)
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return

        # quick tmp size
        try:
            tmp_size = int(tmp.stat().st_size)
        except Exception:
            tmp_size = 0

        # try to acquire global lock (O_EXCL), with timeout
        start = time.time()
        lock_fd = None
        got_lock = False
        while time.time() - start < float(lock_timeout):
            try:
                lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                # write pid for debugging
                try:
                    os.write(lock_fd, str(os.getpid()).encode())
                except Exception:
                    pass
                got_lock = True
                break
            except FileExistsError:
                time.sleep(0.03)
            except Exception:
                break

        if not got_lock:
            # someone else is finalizing — skip installing to avoid blocking
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass
            return

        # lock held: update index and move file atomically
        try:
            # reload index under lock
            self._load_index()

            # enforce size budget
            if self.max_size_bytes is not None:
                # if tmp alone too big -> skip
                if tmp_size > int(self.max_size_bytes):
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    return
                # evict oldest until tmp fits
                self._evict_until_fit(tmp_size)

            # final atomic move
            try:
                os.replace(str(tmp), str(p))
            except Exception:
                try:
                    tmp.rename(p)
                except Exception:
                    # failed to move; cleanup and exit
                    try:
                        if tmp.exists():
                            tmp.unlink()
                    except Exception:
                        pass
                    return

            # update index meta
            try:
                size = int(p.stat().st_size) if p.exists() else tmp_size
            except Exception:
                size = tmp_size
            self._index[p.name] = {"size": size, "atime": time.time()}
            self._save_index()

            # final eviction pass just in case
            if self.max_size_bytes is not None:
                self._evict_until_fit(0)

        finally:
            # release lock and clean up
            try:
                if lock_fd is not None:
                    try:
                        os.close(lock_fd)
                    except Exception:
                        pass
                if lock_path.exists():
                    try:
                        lock_path.unlink()
                    except Exception:
                        pass
            except Exception:
                pass


# -----------------------------
# Optimized Tiling
# -----------------------------

def compute_tile_params(H, W, tile_h, tile_w, stride_h, stride_w):
    """Compute tiling parameters."""
    if H <= tile_h:
        pad_top, pad_bottom = 0, tile_h - H
        n_steps_h = 1
    else:
        n_steps_h = (H - tile_h + stride_h - 1) // stride_h + 1
        full_covered_h = (n_steps_h - 1) * stride_h + tile_h
        pad_top, pad_bottom = 0, max(0, full_covered_h - H)

    if W <= tile_w:
        pad_left, pad_right = 0, tile_w - W
        n_steps_w = 1
    else:
        n_steps_w = (W - tile_w + stride_w - 1) // stride_w + 1
        full_covered_w = (n_steps_w - 1) * stride_w + tile_w
        pad_left, pad_right = 0, max(0, full_covered_w - W)

    return (pad_top, pad_bottom, pad_left, pad_right), n_steps_h, n_steps_w


def extract_tiles_optimized(img, tile_h, tile_w, stride_h, stride_w, pad_mode='reflect'):
    """Ultra-optimized tile extraction using unfold."""
    C, H, W = img.shape
    
    pad_info, n_h, n_w = compute_tile_params(H, W, tile_h, tile_w, stride_h, stride_w)
    pad_top, pad_bottom, pad_left, pad_right = pad_info
    
    # Minimize padding operations
    needs_pad = any(x > 0 for x in pad_info)
    if needs_pad:
        img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode=pad_mode)
    
    # Single unfold operation for all tiles
    tiles = img.unfold(1, tile_h, stride_h).unfold(2, tile_w, stride_w)
    tiles = tiles.permute(1, 2, 0, 3, 4).reshape(-1, C, tile_h, tile_w)
    
    # Vectorized coordinate generation
    ys = torch.arange(n_h) * stride_h - pad_top
    xs = torch.arange(n_w) * stride_w - pad_left
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    
    coords = []
    for i in range(n_h * n_w):
        y, x = yy.flatten()[i].item(), xx.flatten()[i].item()
        y0, x0 = max(0, y), max(0, x)
        y1, x1 = min(H, y + tile_h), min(W, x + tile_w)
        coords.append((y0, y1, x0, x1))
    
    return tiles, coords, (H, W)


def compute_summary_fast(ldr, tiles, coords, orig_shape, tile_h, tile_w, use_half=False):
    """Optimized summary with minimal allocations."""
    H, W = orig_shape
    C, device, dtype = ldr.shape[0], ldr.device, ldr.dtype
    
    # Reuse buffers
    merged = torch.zeros((C, H, W), dtype=dtype, device=device)
    weights = torch.zeros((1, H, W), dtype=dtype, device=device)
    window = make_hann_window(tile_h, tile_w, str(device), use_half)
    
    # Batch process tiles when possible
    for idx, (y0, y1, x0, x1) in enumerate(coords):
        vh, vw = y1 - y0, x1 - x0
        w = window[:vh, :vw].unsqueeze(0)
        # In-place operations
        merged[:, y0:y1, x0:x1].add_(tiles[idx, :, :vh, :vw].mul(w))
        weights[:, y0:y1, x0:x1].add_(w)
    
    merged.div_(weights.clamp_(min=1e-8))
    
    # Fast downsampling
    return F.adaptive_avg_pool2d(merged.unsqueeze(0), (tile_h, tile_w)).squeeze(0)


# -----------------------------
# Smart Sampler - groups tiles from same pair
# -----------------------------
class SmartBatchSampler(data.Sampler):
    """Sampler that groups tiles from same pairs for cache efficiency."""
    
    def __init__(self, tile_index, batch_size, shuffle=True):
        self.tile_index = tile_index
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        # Group indices by pair
        self.pair_groups = defaultdict(list)
        for idx, (pair_idx, tile_idx) in enumerate(tile_index):
            self.pair_groups[pair_idx].append(idx)
    
    def __iter__(self):
        # Shuffle within each pair's tiles
        all_indices = []
        pair_keys = list(self.pair_groups.keys())
        
        if self.shuffle:
            import random
            random.shuffle(pair_keys)
        
        for pair_idx in pair_keys:
            indices = self.pair_groups[pair_idx].copy()
            if self.shuffle:
                import random
                random.shuffle(indices)
            all_indices.extend(indices)
        
        # Yield batches
        for i in range(0, len(all_indices), self.batch_size):
            yield all_indices[i:i + self.batch_size]
    
    def __len__(self):
        return (len(self.tile_index) + self.batch_size - 1) // self.batch_size


# -----------------------------
# Maximum Performance Dataset
# -----------------------------
class HDRDatasetMaxPerf(data.Dataset):
    """Ultimate optimized dataset with all performance features."""
    
    def __init__(self, data_dir, transform=None,
                 tile_h=256, tile_w=256, stride_h=None, stride_w=None,
                 split=None, split_scenes=None,
                 max_cached_pairs=16, use_half=False, 
                 num_load_threads=4, use_disk_cache=False, disk_cache_dir=None,
                 disk_max_size_bytes=(5 * 1024 ** 3), summary_only=True):
        """
        Args:
            use_half: Store tiles in fp16 (50% memory reduction)
            num_load_threads: Parallel image loading threads
            use_disk_cache: Cache preprocessed tiles to disk
            disk_cache_dir: Directory for disk cache
        """
        self.data_dir = data_dir
        self.transform = transform
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.stride_h = stride_h or tile_h
        self.stride_w = stride_w or tile_w
        self.use_half = use_half
        self.max_cached_pairs = max_cached_pairs
        
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # In worker: disable memory cache to save RAM
            # Workers will only use disk cache (shared across workers)
            self.max_cached_pairs = 0
            self._is_worker = True
            print(f"Worker {worker_info.id}: Memory cache disabled (using disk cache only)")
        else:
            # Main process: can use memory cache
            self.max_cached_pairs = max_cached_pairs
            self._is_worker = False
        
        
        # Initialize components
        self.loader = ParallelImageLoader(num_load_threads)
        self.disk_cache = DiskCacheManager(
            disk_cache_dir if use_disk_cache else None,
            max_size_bytes=disk_max_size_bytes,   # set default to 5 GB, or pass via args
            summary_only=summary_only,                # -> very small cache (only sums + tiny tiles)
            downsample_factor=4,
            use_half=self.use_half
        )
        
        # Build pairs
        all_scenes = sorted(os.listdir(data_dir))
        split_scenes_set = set(split_scenes) if split_scenes else None
        self.pairs = []
        
        for scene in all_scenes:
            scene_path = os.path.join(data_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            if split_scenes_set and scene not in split_scenes_set:
                continue
            
            ldr_files = sorted(
                f for f in glob.glob(os.path.join(scene_path, "input_*.tif"))
                if "_aligned" not in os.path.basename(f)
            )
            hdr_files = glob.glob(os.path.join(scene_path, "ref_hdr.hdr"))
            
            if not hdr_files or len(ldr_files) < 3:
                continue
            
            hdr = hdr_files[0]
            self.pairs.append({'ldr1': ldr_files[1], 'ldr2': ldr_files[0], 'hdr': hdr})
            self.pairs.append({'ldr1': ldr_files[1], 'ldr2': ldr_files[2], 'hdr': hdr})
        
        if not self.pairs:
            raise RuntimeError("No valid pairs found")
        
        # Caches
        self._memory_cache = OrderedDict()
        self._tile_index = []
        self._access_count = defaultdict(int)  # Track access frequency
        
        self._build_index()
    
    def _build_index(self):
        """Build index with shape metadata."""
        print("Building high-performance index...")
        
        for pair_idx, pair in enumerate(self.pairs):
            img = cv2.imread(pair['hdr'], cv2.IMREAD_UNCHANGED)
            H, W = img.shape[:2]
            del img  # Free immediately
            
            _, n_h, n_w = compute_tile_params(
                H, W, self.tile_h, self.tile_w, self.stride_h, self.stride_w
            )
            n_tiles = n_h * n_w
            
            for t in range(n_tiles):
                self._tile_index.append((pair_idx, t))
        
        print(f"Index: {len(self._tile_index)} tiles, {len(self.pairs)} pairs")
        if self.use_half:
            print("Using fp16 storage (50% memory reduction)")
    
    def _load_pair_data(self, pair_idx):
        """Load with proper cache hierarchy."""
        
        # 1. Check memory cache (main process only)
        if not self._is_worker and pair_idx in self._memory_cache:
            self._memory_cache.move_to_end(pair_idx)
            self._access_count[pair_idx] += 1
            return self._memory_cache[pair_idx]
        
        # 2. Check disk cache
        tile_params = (self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        cached = self.disk_cache.load(pair_idx, tile_params)
        
        # If we have FULL data in cache, use it
        if cached is not None and 'tiles_g' in cached:
            for k, arr in cached.items():
                if isinstance(arr, np.ndarray):
                    cached[k] = torch.from_numpy(arr)
            
            if not self._is_worker and self.max_cached_pairs > 0:
                self._add_to_memory_cache(pair_idx, cached)
            
            return cached
        
        # 3. If we only have summaries, we must load and recompute tiles
        # (This is expected behavior with summary_only=True)
        pair = self.pairs[pair_idx]
        ldr1, ldr2, hdr = self.loader.load_triplet(
            pair['ldr1'], pair['ldr2'], pair['hdr']
        )
        
        # Convert and process
        t1 = _to_tensor_fast(ldr1, False, self.use_half)
        t2 = _to_tensor_fast(ldr2, False, self.use_half)
        tg = _to_tensor_fast(hdr, True, self.use_half)
        
        if self.transform:
            tg, t1, t2 = self.transform(tg, t1, t2)
        
        # Extract tiles
        tiles_g, coords, shape = extract_tiles_optimized(
            tg, self.tile_h, self.tile_w, self.stride_h, self.stride_w
        )
        tiles1, _, _ = extract_tiles_optimized(
            t1, self.tile_h, self.tile_w, self.stride_h, self.stride_w
        )
        tiles2, _, _ = extract_tiles_optimized(
            t2, self.tile_h, self.tile_w, self.stride_h, self.stride_w
        )
        
        # If we have cached summaries, reuse them (saves computation)
        if cached is not None and 'sum1' in cached and 'sum2' in cached:
            sum1 = cached['sum1']
            sum2 = cached['sum2']
        else:
            # Compute summaries from scratch
            sum1 = compute_summary_fast(t1, tiles1, coords, shape, self.tile_h, self.tile_w, self.use_half)
            sum2 = compute_summary_fast(t2, tiles2, coords, shape, self.tile_h, self.tile_w, self.use_half)
        
        data = {
            'tiles_g': tiles_g,
            'tiles1': tiles1,
            'tiles2': tiles2,
            'sum1': sum1,
            'sum2': sum2
        }
        
        # Save to disk cache (summary only to save disk space)
        self.disk_cache.save(pair_idx, tile_params, data)
        
        # Save to memory cache (main process only)
        if not self._is_worker and self.max_cached_pairs > 0:
            self._add_to_memory_cache(pair_idx, data)
        
        return data
        
    def _add_to_memory_cache(self, pair_idx, data):
        """Add to memory cache with smart eviction."""
        self._memory_cache[pair_idx] = data
        self._memory_cache.move_to_end(pair_idx)
        self._access_count[pair_idx] += 1
        
        # Smart eviction: remove least frequently accessed
        while len(self._memory_cache) > self.max_cached_pairs:
            # Find least accessed
            lru_key = min(self._memory_cache.keys(), 
                         key=lambda k: self._access_count[k])
            self._memory_cache.pop(lru_key)
    
    def __len__(self):
        return len(self._tile_index)
    
    def _clear_worker_cache(self):
        """Clear memory cache in worker processes."""
        if torch.utils.data.get_worker_info() is not None:
            # In worker process - clear cache to save memory
            self._memory_cache.clear()
            self._access_count.clear()

    def __getitem__(self, idx):
        pair_idx, tile_idx = self._tile_index[idx]
        
        # Clear cache periodically in workers
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None and idx % 100 == 0:
            self._clear_worker_cache()
        
        data = self._load_pair_data(pair_idx)
        
        tiles = (
            data['tiles_g'][tile_idx],
            data['tiles1'][tile_idx],
            data['tiles2'][tile_idx],
            data['sum1'],
            data['sum2']
        )
        
        # Convert back to fp32 if needed
        if self.use_half:
            return tuple(t.float() if t.dtype == torch.float16 else t for t in tiles)
        return tiles

    
    def get_smart_sampler(self, batch_size, shuffle=True):
        """Get sampler that groups tiles from same pairs."""
        return SmartBatchSampler(self._tile_index, batch_size, shuffle)
    
    def cleanup(self):
        """Cleanup resources."""
        self.loader.shutdown()
        self._memory_cache.clear()


# Aliases
HDRDatasetTiles = HDRDatasetMaxPerf


# -----------------------------
# CSV helper
# -----------------------------
def read_split_csv(csv_path, split_name):
    """Read CSV with 'scene' and 'split' columns."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {csv_path}")
    
    scenes = set()
    with open(csv_path, newline='', encoding='utf-8') as f:
        # Try comma first
        sample = f.read(1024)
        f.seek(0)
        delimiter = ',' if ',' in sample else '\t'
        
        reader = csv.DictReader(f, delimiter=delimiter, skipinitialspace=True)
        if 'scene' not in reader.fieldnames or 'split' not in reader.fieldnames:
            raise RuntimeError("CSV must contain 'scene' and 'split' columns")
        
        for row in reader:
            scene, sp = row.get('scene', '').strip(), row.get('split', '').strip()
            if scene and sp == split_name:
                scenes.add(scene)
    
    return scenes


def save_tile_png(x, path, percentile=99.0, gamma=2.2):
    """
    Save HDR tensor [C,H,W] to PNG with robust tone mapping.
    """
    img = x.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)
    scale = np.percentile(img, percentile)
    scale = max(scale, 1e-6)
    img = img / scale
    img = np.clip(img, 0.0, 1.0)
    img = img ** (1.0 / gamma)
    img = (img * 255.0).round().astype(np.uint8)

    if img.shape[2] == 3:
        img = img[..., ::-1]

    cv2.imwrite(path, img)
    
    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split_csv", default=None)
    parser.add_argument("--split", choices=['train', 'val', 'all'], default='all')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--tile_h", type=int, default=256)
    parser.add_argument("--tile_w", type=int, default=256)
    parser.add_argument("--stride_h", type=int, default=192)
    parser.add_argument("--stride_w", type=int, default=192)
    parser.add_argument("--max_cached_pairs", type=int, default=16)
    parser.add_argument("--use_half", action='store_true',
                        help="Use fp16 storage (50% memory reduction)")
    parser.add_argument("--num_load_threads", type=int, default=4,
                        help="Parallel image loading threads")
    parser.add_argument("--use_disk_cache", action='store_true',
                        help="Cache preprocessed tiles to disk")
    parser.add_argument("--disk_cache_dir", default="./tile_cache",
                        help="Directory for disk cache")
    parser.add_argument("--use_smart_sampler", action='store_true',
                        help="Use smart sampler for better cache locality")
    args = parser.parse_args()
    
    split_scenes = None
    if args.split_csv and args.split != 'all':
        split_scenes = read_split_csv(args.split_csv, args.split)
        available = set(p.name for p in Path(args.data_dir).iterdir() if p.is_dir())
        split_scenes = sorted(list(split_scenes & available))
        print(f"Using {len(split_scenes)} scenes for '{args.split}'")
    
    ds = HDRDatasetMaxPerf(
        args.data_dir,
        tile_h=args.tile_h, tile_w=args.tile_w,
        stride_h=args.stride_h, stride_w=args.stride_w,
        split=args.split if args.split != 'all' else None,
        split_scenes=split_scenes,
        max_cached_pairs=args.max_cached_pairs,
        use_half=args.use_half,
        num_load_threads=args.num_load_threads,
        use_disk_cache=args.use_disk_cache,
        disk_cache_dir=args.disk_cache_dir if args.use_disk_cache else None
    )
    
    print(f"Dataset: {len(ds)} tiles")
    print(f"Features: half={args.use_half}, threads={args.num_load_threads}, "
          f"disk_cache={args.use_disk_cache}, smart_sampler={args.use_smart_sampler}")
    
    # Use smart sampler if requested
    if args.use_smart_sampler:
        sampler = ds.get_smart_sampler(args.batch_size, shuffle=True)
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=2 if args.num_workers > 0 else None
        )
    else:
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=1 if args.num_workers > 0 else None
        )
    
    print("Testing maximum performance DataLoader...")
    import time
    start = time.time()
    
    os.makedirs("debug_tiles", exist_ok=True)

    for i, (gt, l1, l2, s1, s2) in enumerate(loader):
        if i == 0:
            print(f"Batch shape: gt={gt.shape}, dtype={gt.dtype}")

            # Save first 3 tiles from the batch
            for t in range(min(3, gt.shape[0])):
                save_tile_png(gt[t], f"debug_tiles/gt_{t}.png")
                save_tile_png(l1[t], f"debug_tiles/ldr1_{t}.png")
                save_tile_png(l2[t], f"debug_tiles/ldr2_{t}.png")
                save_tile_png(s1[t], f"debug_tiles/sum1_{t}.png")
                save_tile_png(s2[t], f"debug_tiles/sum2_{t}.png")

            print("Saved sample tiles to ./debug_tiles/")
            break
        
    elapsed = time.time() - start
    print(f"✓ Loaded 10 batches in {elapsed:.2f}s ({10/elapsed:.1f} batches/sec)")
    
    ds.cleanup()