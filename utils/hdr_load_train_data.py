# hdr_load_train_data.py
# FULL VERSION with FIXED cache hashing (stable sha1) so tiles are NOT reprocessed
# + ability to disable persistent tile cache (on-the-fly extraction)

import os
import glob
import math
import pickle
import h5py
import hashlib
from pathlib import Path
from collections import OrderedDict

import numpy as np
import cv2

import torch
import torch.nn.functional as F
import torch.utils.data as data


# -----------------------------
# Utils
# -----------------------------

def _to_tensor_and_normalize(image_np, is_hdr=False):
    """Convert HxWxC numpy image -> torch tensor CxHxW, dtype float32."""
    img = image_np.astype(np.float32)
    if not is_hdr:
        if img.max() > 1.5:
            img = img / 255.0
    if img.ndim == 2:
        img = img[:, :, None]
    img_t = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_t


def make_hann_window(h, w, device=None, dtype=torch.float32):
    if h == 1:
        wh = torch.tensor([1.0], dtype=dtype, device=device)
    else:
        wh = torch.hann_window(h, periodic=False, dtype=dtype, device=device)
    if w == 1:
        ww = torch.tensor([1.0], dtype=dtype, device=device)
    else:
        ww = torch.hann_window(w, periodic=False, dtype=dtype, device=device)
    return wh.unsqueeze(1) @ ww.unsqueeze(0)


# -----------------------------
# Tiling
# -----------------------------

def extract_tiles(img, tile_h, tile_w, stride_h=None, stride_w=None, pad_mode='reflect'):
    """Extract tiles from a torch tensor `img` shaped (C,H,W)."""
    assert isinstance(img, torch.Tensor)
    assert img.ndim == 3

    C, H, W = img.shape
    stride_h = stride_h or tile_h
    stride_w = stride_w or tile_w

    # padding
    if H <= tile_h:
        pad_top, pad_bottom = 0, tile_h - H
    else:
        n_steps_h = math.ceil((H - tile_h) / float(stride_h)) + 1
        full_covered_h = (n_steps_h - 1) * stride_h + tile_h
        pad_top, pad_bottom = 0, max(0, full_covered_h - H)

    if W <= tile_w:
        pad_left, pad_right = 0, tile_w - W
    else:
        n_steps_w = math.ceil((W - tile_w) / float(stride_w)) + 1
        full_covered_w = (n_steps_w - 1) * stride_w + tile_w
        pad_left, pad_right = 0, max(0, full_covered_w - W)

    if any(x > 0 for x in (pad_top, pad_bottom, pad_left, pad_right)):
        img_p = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode=pad_mode)
    else:
        img_p = img

    _, Hp, Wp = img_p.shape
    tiles, coords = [], []
    for y in range(0, Hp - tile_h + 1, stride_h):
        for x in range(0, Wp - tile_w + 1, stride_w):
            tile = img_p[:, y:y + tile_h, x:x + tile_w]
            y0 = max(0, y - pad_top)
            x0 = max(0, x - pad_left)
            y1 = min(H, y - pad_top + tile_h)
            x1 = min(W, x - pad_left + tile_w)
            tiles.append(tile)
            coords.append((y0, y1, x0, x1))

    return tiles, coords, (H, W), (pad_top, pad_bottom, pad_left, pad_right)


def _compute_coords_for_shape(H, W, tile_h, tile_w, stride_h, stride_w):
    """Compute coords list and pad info for an image shape (H,W) without extracting tiles."""
    stride_h = stride_h or tile_h
    stride_w = stride_w or tile_w

    if H <= tile_h:
        pad_top, pad_bottom = 0, tile_h - H
    else:
        n_steps_h = math.ceil((H - tile_h) / float(stride_h)) + 1
        full_covered_h = (n_steps_h - 1) * stride_h + tile_h
        pad_top, pad_bottom = 0, max(0, full_covered_h - H)

    if W <= tile_w:
        pad_left, pad_right = 0, tile_w - W
    else:
        n_steps_w = math.ceil((W - tile_w) / float(stride_w)) + 1
        full_covered_w = (n_steps_w - 1) * stride_w + tile_w
        pad_left, pad_right = 0, max(0, full_covered_w - W)

    Hp = H + pad_top + pad_bottom
    Wp = W + pad_left + pad_right

    coords = []
    for y in range(0, Hp - tile_h + 1, stride_h):
        for x in range(0, Wp - tile_w + 1, stride_w):
            y0 = max(0, y - pad_top)
            x0 = max(0, x - pad_left)
            y1 = min(H, y - pad_top + tile_h)
            x1 = min(W, x - pad_left + tile_w)
            coords.append((y0, y1, x0, x1))

    return coords, (pad_top, pad_bottom, pad_left, pad_right)


# -----------------------------
# Dataset
# -----------------------------
class HDRDatasetTiles(data.Dataset):
    def __init__(self, data_dir, use_log=True, transform=None,
                 tile_h=None, tile_w=None, stride_h=None, stride_w=None,
                 cache_dir=None, use_hdf5=True, use_worker_cache=False,
                 use_persistent_cache=True):
        """
        use_persistent_cache: if False, tiles will NOT be stored on disk.
                              Tiles will be computed on-the-fly per __getitem__.
        """

        self.data_dir = data_dir
        self.use_log = use_log
        self.transform = transform
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.stride_h = stride_h or tile_h
        self.stride_w = stride_w or tile_w
        self.use_hdf5 = use_hdf5
        self.use_worker_cache = use_worker_cache
        self.use_persistent_cache = use_persistent_cache

        # scenes / pairs
        self.scenes = sorted(os.listdir(data_dir))
        self.pairs = []
        for scene in self.scenes:
            scene_path = os.path.join(data_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            ldr_files = sorted(
                f for f in glob.glob(os.path.join(scene_path, "input_*.tif"))
                if "_aligned" not in os.path.basename(f)
            )
            hdr_files = glob.glob(os.path.join(scene_path, "ref_hdr.hdr"))
            if not hdr_files:
                continue
            hdr = hdr_files[0]

            # guard if less than 3 ldr inputs (avoid index error)
            if len(ldr_files) < 3:
                # skip or use available combinations sensibly
                continue

            self.pairs.append({
                'ldr1': ldr_files[1],
                'ldr2': ldr_files[0],
                'hdr': hdr
            })

            # second pair (1,2)
            if len(ldr_files) >= 3:
                self.pairs.append({
                    'ldr1': ldr_files[1],
                    'ldr2': ldr_files[2],
                    'hdr': hdr
                })

        if not self.pairs:
            raise RuntimeError("No valid pairs found")

        # cache paths (only create if using persistent cache)
        self.cache_dir = None
        if self.use_persistent_cache:
            self.cache_dir = Path(cache_dir) if cache_dir else Path(data_dir) / "tile_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.metadata_file = self.cache_dir / "tile_metadata.pkl"
        else:
            self.metadata_file = None

        self.tiles_index = []
        # pair_coords used when not using persistent cache
        self.pair_coords = {}

        if self.tile_h is not None:
            if self.use_persistent_cache:
                self._load_or_build_tile_cache()
            else:
                # build an in-memory index of coords per pair (no disk writes)
                self._build_in_memory_index()

        self._summary_cache = OrderedDict() if use_worker_cache else None

    # -----------------------------
    # Stable hash (FIX)
    # -----------------------------
    def _compute_config_hash(self):
        m = hashlib.sha1()
        for p in sorted(os.path.abspath(x['ldr1']) for x in self.pairs):
            m.update(p.encode())
            m.update(b'\0')
        m.update(f"{self.tile_h},{self.tile_w},{self.stride_h},{self.stride_w},{self.use_log}".encode())
        return m.hexdigest()

    def _get_hdf5_path(self):
        return self.cache_dir / "tiles.h5" if self.cache_dir is not None else None

    def _get_tile_dir(self, pair_idx):
        return self.cache_dir / f"pair_{pair_idx:06d}" if self.cache_dir is not None else None

    # -----------------------------
    # Cache handling
    # -----------------------------
    def _load_or_build_tile_cache(self):
        cfg_hash = self._compute_config_hash()

        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'rb') as f:
                    meta = pickle.load(f)
                if meta.get('config_hash') == cfg_hash:
                    self.tiles_index = meta['tiles_index']
                    if self.use_hdf5:
                        h5_path = self._get_hdf5_path()
                        if h5_path and h5_path.exists():
                            print(f"Loaded tile cache ({len(self.tiles_index)} tiles)")
                            return
                    else:
                        pair_dirs = set(pi for pi, _ in self.tiles_index)
                        if all(self._get_tile_dir(pi).exists() for pi in pair_dirs):
                            print(f"Loaded tile cache ({len(self.tiles_index)} tiles)")
                            return
            except Exception as e:
                print("Failed to load cache metadata:", e)

        print("Pre-extracting tiles to disk (one-time operation)...")
        self._build_tile_cache()

    def _build_tile_cache(self):
        self.tiles_index = []

        if self.use_hdf5:
            h5_path = self._get_hdf5_path()
            if h5_path.exists():
                h5_path.unlink()

            with h5py.File(h5_path, 'w') as f:
                for pair_idx, pair in enumerate(self.pairs):
                    n_tiles = self._extract_pair_to_hdf5(f, pair_idx, pair)
                    for t in range(n_tiles):
                        self.tiles_index.append((pair_idx, t))
                    if (pair_idx + 1) % 10 == 0:
                        print(f"  Processed {pair_idx + 1}/{len(self.pairs)} pairs")
        else:
            for pair_idx, pair in enumerate(self.pairs):
                n_tiles = self._extract_pair_to_files(pair_idx, pair)
                for t in range(n_tiles):
                    self.tiles_index.append((pair_idx, t))

        meta = {
            'config_hash': self._compute_config_hash(),
            'tiles_index': self.tiles_index
        }
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(meta, f)

        print(f"Tile cache built: {len(self.tiles_index)} tiles")

    def _build_in_memory_index(self):
        """Create pair_coords and tiles_index without writing tiles to disk."""
        self.tiles_index = []
        self.pair_coords = {}
        for pair_idx, pair in enumerate(self.pairs):
            # read shape quickly (use hdr image as reference for shape)
            img = cv2.imread(pair['hdr'], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read image for pair {pair_idx}")
            H, W = img.shape[0], img.shape[1]
            coords, pad_info = _compute_coords_for_shape(H, W, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
            self.pair_coords[pair_idx] = (coords, (H, W))
            for t in range(len(coords)):
                self.tiles_index.append((pair_idx, t))
        print(f"Built in-memory tile index: {len(self.tiles_index)} tiles (no persistent cache)")

    # -----------------------------
    # Extraction
    # -----------------------------
    def _load_images(self, pair):
        ldr1 = cv2.imread(pair['ldr1'], cv2.IMREAD_UNCHANGED)
        ldr2 = cv2.imread(pair['ldr2'], cv2.IMREAD_UNCHANGED)
        hdr = cv2.imread(pair['hdr'], cv2.IMREAD_UNCHANGED)
        for x in (ldr1, ldr2, hdr):
            if x is None:
                raise RuntimeError("Failed to read image")
        if ldr1.ndim == 3:
            ldr1 = cv2.cvtColor(ldr1, cv2.COLOR_BGR2RGB)
            ldr2 = cv2.cvtColor(ldr2, cv2.COLOR_BGR2RGB)
        if hdr.ndim == 3:
            hdr = cv2.cvtColor(hdr, cv2.COLOR_BGR2RGB)
        t1 = _to_tensor_and_normalize(ldr1, False)
        t2 = _to_tensor_and_normalize(ldr2, False)
        tg = _to_tensor_and_normalize(hdr, True)
        if self.use_log:
            tg = torch.log1p(torch.clamp(tg, min=0))
        if self.transform:
            tg, t1, t2 = self.transform(tg, t1, t2)
        return tg, t1, t2

    def _compute_summary(self, ldr1, ldr2, tiles1, tiles2, coords, orig_shape):
        H, W = orig_shape
        win = make_hann_window(self.tile_h, self.tile_w, device=ldr1.device, dtype=ldr1.dtype)

        def merge(tiles, base):
            out = torch.zeros_like(base)
            wgt = torch.zeros((1, H, W), dtype=base.dtype)
            for tile, (y0, y1, x0, x1) in zip(tiles, coords):
                vh, vw = y1 - y0, x1 - x0
                ws = win[:vh, :vw].unsqueeze(0)
                out[:, y0:y1, x0:x1] += tile[:, :vh, :vw] * ws
                wgt[:, y0:y1, x0:x1] += ws
            return out / (wgt + 1e-8)

        s1 = merge(tiles1, ldr1)
        s2 = merge(tiles2, ldr2)
        s1 = F.adaptive_avg_pool2d(s1.unsqueeze(0), (self.tile_h, self.tile_w)).squeeze(0)
        s2 = F.adaptive_avg_pool2d(s2.unsqueeze(0), (self.tile_h, self.tile_w)).squeeze(0)
        return s1, s2

    def _extract_pair_to_hdf5(self, h5f, pair_idx, pair):
        tg, t1, t2 = self._load_images(pair)
        tiles_g, coords, orig_shape, _ = extract_tiles(tg, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        tiles1, _, _, _ = extract_tiles(t1, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        tiles2, _, _, _ = extract_tiles(t2, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        sum1, sum2 = self._compute_summary(t1, t2, tiles1, tiles2, coords, orig_shape)

        grp = h5f.create_group(f"pair_{pair_idx:06d}")
        grp.create_dataset('gt', data=torch.stack(tiles_g).numpy(), compression='gzip', compression_opts=1)
        grp.create_dataset('ldr1', data=torch.stack(tiles1).numpy(), compression='gzip', compression_opts=1)
        grp.create_dataset('ldr2', data=torch.stack(tiles2).numpy(), compression='gzip', compression_opts=1)
        grp.create_dataset('sum1', data=sum1.numpy(), compression='gzip', compression_opts=1)
        grp.create_dataset('sum2', data=sum2.numpy(), compression='gzip', compression_opts=1)
        return len(tiles_g)

    def _extract_pair_to_files(self, pair_idx, pair):
        tg, t1, t2 = self._load_images(pair)
        tiles_g, coords, orig_shape, _ = extract_tiles(tg, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        tiles1, _, _, _ = extract_tiles(t1, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        tiles2, _, _, _ = extract_tiles(t2, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        sum1, sum2 = self._compute_summary(t1, t2, tiles1, tiles2, coords, orig_shape)

        d = self._get_tile_dir(pair_idx)
        d.mkdir(parents=True, exist_ok=True)
        for i in range(len(tiles_g)):
            torch.save({'gt': tiles_g[i], 'ldr1': tiles1[i], 'ldr2': tiles2[i], 'sum1': sum1, 'sum2': sum2}, d / f"tile_{i:04d}.pt")
        return len(tiles_g)

    # -----------------------------
    # Dataset API
    # -----------------------------
    def __len__(self):
        return len(self.tiles_index)

    def __getitem__(self, idx):
        pair_idx, tile_idx = self.tiles_index[idx]

        # Persistent HDF5 / file-backed cache path (unchanged)
        if self.use_persistent_cache:
            if self.use_hdf5:
                with h5py.File(self._get_hdf5_path(), 'r') as f:
                    g = f[f"pair_{pair_idx:06d}"]
                    return (
                        torch.from_numpy(g['gt'][tile_idx]).float(),
                        torch.from_numpy(g['ldr1'][tile_idx]).float(),
                        torch.from_numpy(g['ldr2'][tile_idx]).float(),
                        torch.from_numpy(g['sum1'][:]).float(),
                        torch.from_numpy(g['sum2'][:]).float(),
                    )
            else:
                d = torch.load(self._get_tile_dir(pair_idx) / f"tile_{tile_idx:04d}.pt")
                return d['gt'], d['ldr1'], d['ldr2'], d['sum1'], d['sum2']

        # On-the-fly extraction path (no persistent cache)
        # We compute tiles for the requested pair and return the requested tile.
        pair = self.pairs[pair_idx]
        # optionally reuse computed summary for this worker/pair
        if self._summary_cache is not None and pair_idx in self._summary_cache:
            sum1, sum2 = self._summary_cache[pair_idx]
        else:
            # load images and extract tiles (we keep tiles lists in local scope only)
            tg, t1, t2 = self._load_images(pair)
            tiles_g, coords, orig_shape, _ = extract_tiles(tg, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
            tiles1, _, _, _ = extract_tiles(t1, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
            tiles2, _, _, _ = extract_tiles(t2, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
            sum1, sum2 = self._compute_summary(t1, t2, tiles1, tiles2, coords, orig_shape)
            if self._summary_cache is not None:
                # keep small cache keyed by pair index
                self._summary_cache[pair_idx] = (sum1, sum2)

            return tiles_g[tile_idx], tiles1[tile_idx], tiles2[tile_idx], sum1, sum2

        # if summary came from cache we still need the specific tile: load images and crop the tile
        tg, t1, t2 = self._load_images(pair)
        tiles_g, coords, orig_shape, _ = extract_tiles(tg, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        tiles1, _, _, _ = extract_tiles(t1, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        tiles2, _, _, _ = extract_tiles(t2, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        sum1, sum2 = self._summary_cache[pair_idx]
        return tiles_g[tile_idx], tiles1[tile_idx], tiles2[tile_idx], sum1, sum2


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--use_hdf5", action="store_true", default=True)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_persistent_cache", action="store_true",
                        help="Disable persistent disk cache; extract tiles on-the-fly (slower).")
    args = parser.parse_args()

    use_persistent_cache = not args.no_persistent_cache

    if args.preprocess:
        # Instantiate dataset with persistent cache enabled to build the cache
        ds = HDRDatasetTiles(
            args.data_dir,
            tile_h=256, tile_w=256,
            stride_h=192, stride_w=192,
            cache_dir=args.cache_dir,
            use_hdf5=args.use_hdf5,
            use_worker_cache=False,
            use_persistent_cache=use_persistent_cache
        )
        print("Preprocessing complete (cache built or in-memory index created).")
    else:
        # Normal training usage
        ds = HDRDatasetTiles(
            args.data_dir,
            tile_h=256, tile_w=256,
            stride_h=192, stride_w=192,
            cache_dir=args.cache_dir,
            use_hdf5=args.use_hdf5,
            use_worker_cache=False,  # Disable worker cache to save memory
            use_persistent_cache=use_persistent_cache
        )

        print(f"Dataset length: {len(ds)} tiles")
        print(f"Creating DataLoader with {args.num_workers} workers...")

        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=True if args.num_workers > 0 else False,
            prefetch_factor=2 if args.num_workers > 0 else None
        )

        print("Testing DataLoader...")
        for i, (gt, l1, l2, s1, s2) in enumerate(loader):
            print(f"Batch {i}: gt={gt.shape}, ldr1={l1.shape}, ldr2={l2.shape}, "
                  f"sum1={s1.shape}, sum2={s2.shape}")
            if i >= 2:
                break

        print("Success! No memory issues.")
