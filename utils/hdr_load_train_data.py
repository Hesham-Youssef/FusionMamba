import os
import cv2
from glob import glob
import numpy as np
from torch.utils.data import Dataset
from utils.tools import get_image, LDR2HDR_batch, LDR2HDR


def create_global_summary(full_image, target_size):
    h, w, c = full_image.shape
    target_h, target_w = target_size
    
    # Use area interpolation (best for downsampling)
    summary = cv2.resize(full_image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    
    return summary


class U2NetDataset(Dataset):
    """
    Dataset for U2Net HDR reconstruction.
    - Each image gets its OWN global summary
    - Randomly samples pairs of exposures for training diversity
    - No caching to avoid OOM
    """
    def __init__(self, configs,
                 input_name='input_*_aligned.tif',
                 ref_name='ref_*_aligned.tif',
                 input_exp_name='input_exp.txt',
                 ref_exp_name='ref_exp.txt',
                 ref_hdr_name='ref_hdr_aligned.hdr'):
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
        
        # Count patches per scene
        self.count = []
        for i, scene_dir in enumerate(self.scene_dirs):
            cur_scene_dir = os.path.join(self.filepath, scene_dir)
            in_LDR_paths = sorted(glob(os.path.join(cur_scene_dir, input_name)))
            tmp_img = get_image(in_LDR_paths[0]).astype(np.float32)
            h, w, c = tmp_img.shape
            if h < self.patch_size[0] or w < self.patch_size[1]:
                raise AttributeError('Training images smaller than patch size.')
            h_count = np.ceil(h / self.patch_stride)
            w_count = np.ceil(w / self.patch_stride)
            self.count.append(h_count * w_count)
            self.total_count = self.total_count + h_count * w_count
        self.count = np.array(self.count).astype(int)
        self.total_count = int(self.total_count)

        print(f'====> Finish preparing U2Net training data! Total patches: {self.total_count}')

    def __len__(self):
        return self.total_count

    def __getitem__(self, index):
        # Find corresponding scene
        idx_beg = 0
        cur_scene_dir = ""
        scene_idx = -1
        scene_posidx = -1
        for i, scene_dir in enumerate(self.scene_dirs):
            idx_end = idx_beg + self.count[i]
            if idx_beg <= index < idx_end:
                cur_scene_dir = os.path.join(self.filepath, scene_dir)
                scene_idx = i
                scene_posidx = index - idx_beg
                break
            idx_beg = idx_end
        if scene_idx == -1:
            raise ValueError('Index out of bound')

        scene_dir = self.scene_dirs[scene_idx]
        
        # Load image paths
        in_LDR_paths = sorted(glob(os.path.join(cur_scene_dir, self.input_name)))
        
        # Load exposures
        in_exp_path = os.path.join(cur_scene_dir, self.input_exp_name)
        in_exp = np.array(open(in_exp_path).read().split('\n')[:self.num_shots]).astype(np.float32)
        in_exp = 2 ** in_exp
        
        # Get image dimensions from first image
        tmp_img = get_image(in_LDR_paths[0])
        h, w, c = tmp_img.shape

        # Calculate patch coordinates
        h_count = np.ceil(h / self.patch_stride)
        w_count = np.ceil(w / self.patch_stride)
        h_idx = int(scene_posidx / w_count)
        w_idx = int(scene_posidx - h_idx * w_count)

        h_up = h_idx * self.patch_stride
        h_down = h_idx * self.patch_stride + self.patch_size[0]
        if h_down > h:
            h_up = h - self.patch_size[0]
            h_down = h

        w_left = w_idx * self.patch_stride
        w_right = w_idx * self.patch_stride + self.patch_size[1]
        if w_right > w:
            w_left = w - self.patch_size[1]
            w_right = w

        # ================================================================
        # EXPOSURE PAIR SELECTION:
        # - img1 = ALWAYS middle exposure (motion reference)
        # - img2 = randomly selected from OTHER exposures
        # ================================================================
        # Middle exposure is the motion reference
        idx1 = self.num_shots // 2
        
        if self.num_shots >= 2:
            # Select img2 from all exposures EXCEPT the middle one
            other_indices = [i for i in range(self.num_shots) if i != idx1]
            idx2 = np.random.choice(other_indices)
        else:
            idx2 = 0  # Fallback if only one exposure
        
        # ================================================================
        # Load ONLY the two selected exposures (no full image loading)
        # ================================================================
        
        # Load patches for the two selected exposures
        img1_ldr_patch = get_image(in_LDR_paths[idx1])[h_up:h_down, w_left:w_right, :]
        img2_ldr_patch = get_image(in_LDR_paths[idx2])[h_up:h_down, w_left:w_right, :]
        
        # Convert to HDR (use LDR2HDR for single exposure, not LDR2HDR_batch)
        img1_hdr_patch = LDR2HDR(img1_ldr_patch, in_exp[idx1])
        img2_hdr_patch = LDR2HDR(img2_ldr_patch, in_exp[idx2])
        
        # Load FULL images for summaries (needed for global context)
        img1_ldr_full = get_image(in_LDR_paths[idx1])
        img2_ldr_full = get_image(in_LDR_paths[idx2])
        
        # Convert to HDR (use LDR2HDR for single exposure)
        img1_hdr_full = LDR2HDR(img1_ldr_full, in_exp[idx1])
        img2_hdr_full = LDR2HDR(img2_ldr_full, in_exp[idx2])

        # Load reference HDR (patch only)
        ref_HDR = get_image(os.path.join(cur_scene_dir, self.ref_hdr_name))[h_up:h_down, w_left:w_right, :]

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
            
            img1_ldr_full = np.flip(img1_ldr_full, axis=1)
            img2_ldr_full = np.flip(img2_ldr_full, axis=1)
            img1_hdr_full = np.flip(img1_hdr_full, axis=1)
            img2_hdr_full = np.flip(img2_hdr_full, axis=1)

        # Rotation
        k = int(distortions[1] * 4 + 0.5)
        img1_ldr_patch = np.rot90(img1_ldr_patch, k)
        img2_ldr_patch = np.rot90(img2_ldr_patch, k)
        img1_hdr_patch = np.rot90(img1_hdr_patch, k)
        img2_hdr_patch = np.rot90(img2_hdr_patch, k)
        ref_HDR = np.rot90(ref_HDR, k)
        
        img1_ldr_full = np.rot90(img1_ldr_full, k)
        img2_ldr_full = np.rot90(img2_ldr_full, k)
        img1_hdr_full = np.rot90(img1_hdr_full, k)
        img2_hdr_full = np.rot90(img2_hdr_full, k)
        
        # ================================================================
        # Create GLOBAL summaries (one for EACH exposure)
        # Downsample to patch size
        # ================================================================
        sum1_ldr = create_global_summary(img1_ldr_full, self.patch_size)
        sum1_hdr = create_global_summary(img1_hdr_full, self.patch_size)
        sum2_ldr = create_global_summary(img2_ldr_full, self.patch_size)
        sum2_hdr = create_global_summary(img2_hdr_full, self.patch_size)
        
        # Concatenate LDR and HDR for each summary (6 channels: 3 LDR + 3 HDR)
        # This matches img1 = cat([img1_ldr, img1_hdr]) which is also 6 channels
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
    Test dataset for U2Net HDR reconstruction.
    - Each image gets its OWN global summary
    - Uses first two exposures consistently (no randomness in test)
    """
    def __init__(self, configs,
                 input_name='input_*_aligned.tif',
                 input_exp_name='input_exp.txt',
                 ref_hdr_name='ref_hdr_aligned.hdr'):
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
        
        print('====> Finish preparing U2Net testing data!')

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
        
        # Use middle exposure as img1 (motion reference), first exposure as img2
        idx1 = ns // 2  # Middle exposure
        idx2 = 0 if idx1 != 0 else 1  # First exposure (or second if middle is first)
        
        # Load the two exposures
        img1_ldr = get_image(LDR_paths[idx1], image_size=[h, w], is_crop=True)
        img2_ldr = get_image(LDR_paths[idx2], image_size=[h, w], is_crop=True)
        
        img1_hdr = LDR2HDR(img1_ldr, 2. ** in_exps[idx1])
        img2_hdr = LDR2HDR(img2_ldr, 2. ** in_exps[idx2])

        # Load reference HDR
        ref_HDR_path = os.path.join(scene_path, self.ref_hdr_name)
        ref_HDR = get_image(ref_HDR_path, image_size=[h, w], is_crop=True)

        # Create GLOBAL summaries (one for EACH exposure)
        # Downsample full images to patch size
        sum1_ldr = create_global_summary(img1_ldr, self.patch_size)
        sum1_hdr = create_global_summary(img1_hdr, self.patch_size)
        sum2_ldr = create_global_summary(img2_ldr, self.patch_size)
        sum2_hdr = create_global_summary(img2_hdr, self.patch_size)
        
        # Concatenate LDR and HDR for each (6 channels total)
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