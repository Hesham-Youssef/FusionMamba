# hdr_dataset_consistent.py
import torch
import torch.utils.data as data
import cv2
import os
import numpy as np

class HDRDataset(data.Dataset):
    """
    Returns (gt_hdr, ldr_short, ldr_long)
    - gt_hdr:   (C, target_H, target_W)      [log1p if use_log=True]
    - ldr_long: (C, target_H, target_W)      (one LDR exposure, crop-resized)
    - ldr_short:(C, target_H/ratio, target_W/ratio) (other LDR exposure, same crop then downsampled)
    """
    def __init__(self, data_dir, target_H=64, target_W=64, ratio=4,
                 use_log=True, random_crop=True, transform=None):
        super(HDRDataset, self).__init__()
        assert target_H % ratio == 0 and target_W % ratio == 0, \
            "target_H and target_W must be divisible by ratio"
        self.data_dir = data_dir
        self.scenes = sorted(os.listdir(data_dir))
        self.use_log = use_log
        self.transform = transform
        self.target_H = target_H
        self.target_W = target_W
        self.ratio = ratio
        self.random_crop = random_crop

        self.pairs = []
        for scene in self.scenes:
            scene_path = os.path.join(data_dir, scene)
            if not os.path.isdir(scene_path):
                continue
            ldr_files = sorted([f for f in os.listdir(scene_path) if f.lower().endswith('.tif')])
            hdr_list = [f for f in os.listdir(scene_path) if f.lower().endswith('.hdr')]
            if len(hdr_list) == 0:
                raise RuntimeError(f"No .hdr file found in {scene_path}")
            hdr_file = hdr_list[0]
            for i in range(len(ldr_files) - 1):
                self.pairs.append({
                    'scene_path': scene_path,
                    'ldr1': ldr_files[i],
                    'ldr2': ldr_files[i+1],
                    'hdr': hdr_file
                })

    def __len__(self):
        return len(self.pairs)

    def _consistent_crop_or_resize(self, imgs):
        """
        imgs: list of numpy arrays [ldr1, ldr2, gt_hdr] in original sizes
        returns: list of images where ldr_long and gt_hdr are (target_H, target_W)
                 and ldr_short will be produced separately by downsampling below.
        This function returns the *same spatial crop/resized* images for all inputs.
        """
        h0, w0 = imgs[0].shape[:2]
        th, tw = self.target_H, self.target_W

        if self.random_crop and h0 >= th and w0 >= tw:
            top = np.random.randint(0, h0 - th + 1)
            left = np.random.randint(0, w0 - tw + 1)
            return [img[top:top+th, left:left+tw].copy() for img in imgs]
        else:
            return [cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR) for img in imgs]

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        scene_path = pair['scene_path']

        ldr1 = cv2.imread(os.path.join(scene_path, pair['ldr1']), cv2.IMREAD_UNCHANGED)
        ldr2 = cv2.imread(os.path.join(scene_path, pair['ldr2']), cv2.IMREAD_UNCHANGED)
        gt_hdr = cv2.imread(os.path.join(scene_path, pair['hdr']), -1)  # float32 HDR
        
        if ldr1 is None or ldr2 is None or gt_hdr is None:
            raise RuntimeError(f"Failed to read files in {scene_path}: {pair}")

        # Convert to float
        ldr1 = ldr1.astype(np.float32) / 255.0
        ldr2 = ldr2.astype(np.float32) / 255.0

        # Convert BGR -> RGB if 3-channel
        if ldr1.ndim == 3 and ldr1.shape[2] == 3:
            ldr1 = cv2.cvtColor(ldr1, cv2.COLOR_BGR2RGB)
            ldr2 = cv2.cvtColor(ldr2, cv2.COLOR_BGR2RGB)
        if gt_hdr.ndim == 3 and gt_hdr.shape[2] == 3:
            gt_hdr = cv2.cvtColor(gt_hdr, cv2.COLOR_BGR2RGB)

        # Randomly swap exposures (same behavior as before)
        if np.random.rand() > 0.5:
            ldr1, ldr2 = ldr2, ldr1

        # Crop or resize all three consistently to (target_H, target_W)
        ldr1_c, ldr2_c, gt_c = self._consistent_crop_or_resize([ldr1, ldr2, gt_hdr])

        # ldr_long: use ldr1_c resized/cropped -> (target_H, target_W)
        ldr_long = ldr1_c

        # ldr_short: downsample ldr2_c from (target_H,target_W) -> (target_H/ratio, target_W/ratio)
        sh, sw = self.target_H // self.ratio, self.target_W // self.ratio
        ldr_short = cv2.resize(ldr2_c, (sw, sh), interpolation=cv2.INTER_AREA)

        # gt: optionally log-space
        if self.use_log:
            gt_c = np.log1p(gt_c)

        # To tensors (C,H,W)
        ldr_short = torch.from_numpy(ldr_short).permute(2,0,1).float()
        ldr_long  = torch.from_numpy(ldr_long).permute(2,0,1).float()
        gt_hdr    = torch.from_numpy(gt_c).permute(2,0,1).float()

        # Optional transform
        if self.transform:
            ldr_short, ldr_long, gt_hdr = self.transform(ldr_short, ldr_long, gt_hdr)

        return gt_hdr, ldr_short, ldr_long
