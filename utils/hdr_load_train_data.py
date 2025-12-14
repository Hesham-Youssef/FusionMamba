# hdr_load_train_data.py
# Simplified version with on-the-fly tile extraction only

import os
import glob
import math

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
                 split=None, split_scenes=None):
        """
        split: 'train', 'val', or None
        split_scenes: list of scene names to include in this split (optional)
        """

        self.data_dir = data_dir
        self.use_log = use_log
        self.transform = transform
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.stride_h = stride_h or tile_h
        self.stride_w = stride_w or tile_w
        self.split = split
        self.split_scenes = set(split_scenes) if split_scenes is not None else None

        # scenes / pairs
        all_scenes = sorted(os.listdir(data_dir))
        self.scenes = []
        self.pairs = []
        for scene in all_scenes:
            if not os.path.isdir(os.path.join(data_dir, scene)):
                continue
            # Apply split filter if provided
            if self.split_scenes is not None and scene not in self.split_scenes:
                continue
            self.scenes.append(scene)
            scene_path = os.path.join(data_dir, scene)
            ldr_files = sorted(
                f for f in glob.glob(os.path.join(scene_path, "input_*.tif"))
                if "_aligned" not in os.path.basename(f)
            )
            hdr_files = glob.glob(os.path.join(scene_path, "ref_hdr.hdr"))
            if not hdr_files or len(ldr_files) < 2:
                continue
            hdr = hdr_files[0]
            # create pairs
            self.pairs.append({'ldr1': ldr_files[1], 'ldr2': ldr_files[0], 'hdr': hdr})
            if len(ldr_files) >= 3:
                self.pairs.append({'ldr1': ldr_files[1], 'ldr2': ldr_files[2], 'hdr': hdr})

        if not self.pairs:
            raise RuntimeError(f"No valid pairs found for split={split}")

        self.tiles_index = []

        if self.tile_h is not None:
            self._build_in_memory_index()

    def _build_in_memory_index(self):
        """Create tiles_index for accessing individual tiles."""
        self.tiles_index = []
        for pair_idx, pair in enumerate(self.pairs):
            # read shape quickly (use hdr image as reference for shape)
            img = cv2.imread(pair['hdr'], cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to read image for pair {pair_idx}")
            H, W = img.shape[0], img.shape[1]
            coords, pad_info = _compute_coords_for_shape(H, W, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
            for t in range(len(coords)):
                self.tiles_index.append((pair_idx, t))
        print(f"Built in-memory tile index: {len(self.tiles_index)} tiles")

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

    def __len__(self):
        return len(self.tiles_index)

    def __getitem__(self, idx):
        pair_idx, tile_idx = self.tiles_index[idx]
        pair = self.pairs[pair_idx]

        # Load images and extract tiles on-the-fly
        tg, t1, t2 = self._load_images(pair)
        tiles_g, coords, orig_shape, _ = extract_tiles(tg, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        tiles1, _, _, _ = extract_tiles(t1, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        tiles2, _, _, _ = extract_tiles(t2, self.tile_h, self.tile_w, self.stride_h, self.stride_w)
        sum1, sum2 = self._compute_summary(t1, t2, tiles1, tiles2, coords, orig_shape)

        return tiles_g[tile_idx], tiles1[tile_idx], tiles2[tile_idx], sum1, sum2


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    # Create dataset with on-the-fly tile extraction
    ds = HDRDatasetTiles(
        args.data_dir,
        tile_h=256, tile_w=256,
        stride_h=192, stride_w=192
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

    print("Success! DataLoader working correctly.")