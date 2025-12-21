import os
import cv2
from glob import glob
import numpy as np
from torch.utils.data import Dataset
from utils.tools import get_image
from utils.tools import LDR2HDR_batch

class U2NetDataset(Dataset):
    """
    Dataset for U2Net HDR reconstruction
    Prepares paired LDR images with their summaries for fusion
    """
    def __init__(self, configs,
                 input_name='input_*_aligned.tif',
                 ref_name='ref_*_aligned.tif',
                 input_exp_name='input_exp.txt',
                 ref_exp_name='ref_exp.txt',
                 ref_hdr_name='ref_hdr_aligned.hdr'):
        super().__init__()
        print('====> Start preparing U2Net training data.')

        # Basic information
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

        in_LDR_paths = sorted(glob(os.path.join(cur_scene_dir, self.input_name)))
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

        # Load input LDR images
        in_LDR = np.zeros((self.patch_size[0], self.patch_size[1], c * self.num_shots))
        for j, in_LDR_path in enumerate(in_LDR_paths):
            in_LDR[:, :, j * c:(j + 1) * c] = get_image(in_LDR_path)[h_up:h_down, w_left:w_right, :]
        in_LDR = np.array(in_LDR).astype(np.float32)

        # Load exposures
        in_exp_path = os.path.join(cur_scene_dir, self.input_exp_name)
        in_exp = np.array(open(in_exp_path).read().split('\n')[:self.num_shots]).astype(np.float32)

        # Load reference HDR
        ref_HDR = get_image(os.path.join(cur_scene_dir, self.ref_hdr_name))[h_up:h_down, w_left:w_right, :]

        # Apply random augmentations
        distortions = np.random.uniform(0.0, 1.0, 2)
        
        # Horizontal flip
        if distortions[0] < 0.5:
            in_LDR = np.flip(in_LDR, axis=1)
            ref_HDR = np.flip(ref_HDR, axis=1)

        # Rotation
        k = int(distortions[1] * 4 + 0.5)
        in_LDR = np.rot90(in_LDR, k)
        ref_HDR = np.rot90(ref_HDR, k)
        
        # Convert exposures
        in_exp = 2 ** in_exp

        # Convert LDR to HDR
        in_HDR = LDR2HDR_batch(in_LDR, in_exp)

        # Prepare pairs for U2Net: use first two exposures as img1 and img2
        # img1: first exposure (LDR + HDR)
        # img2: second exposure (LDR + HDR)
        # sum1: average of all LDR images
        # sum2: average of all HDR images
        
        img1_ldr = in_LDR[:, :, 0:c]
        img2_ldr = in_LDR[:, :, c:c*2]
        img1_hdr = in_HDR[:, :, 0:c]
        img2_hdr = in_HDR[:, :, c:c*2]
        
        # Create summaries (mean across all exposures) - concatenate LDR and HDR
        sum_ldr = np.mean(in_LDR.reshape(self.patch_size[0], self.patch_size[1], self.num_shots, c), axis=2)
        sum_hdr = np.mean(in_HDR.reshape(self.patch_size[0], self.patch_size[1], self.num_shots, c), axis=2)
        
        # Concatenate to match img1/img2 structure (6 channels total)
        sum1 = np.concatenate([sum_ldr, sum_hdr], axis=2)
        sum2 = np.concatenate([sum_ldr, sum_hdr], axis=2)
        
        # Transpose to PyTorch format (C, H, W)
        img1_ldr = np.einsum("ijk->kij", img1_ldr)
        img2_ldr = np.einsum("ijk->kij", img2_ldr)
        img1_hdr = np.einsum("ijk->kij", img1_hdr)
        img2_hdr = np.einsum("ijk->kij", img2_hdr)
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
    Test dataset for U2Net HDR reconstruction
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
        
        # Load all LDR images
        in_LDRs = np.zeros((h, w, c * ns), dtype=np.float32)
        in_HDRs = np.zeros((h, w, c * ns), dtype=np.float32)

        for i, image_path in enumerate(LDR_paths):
            img = get_image(image_path, image_size=[h, w], is_crop=True)
            in_LDRs[:, :, c * i:c * (i + 1)] = img
            from utils.tools import LDR2HDR
            in_HDRs[:, :, c * i:c * (i + 1)] = LDR2HDR(img, 2. ** in_exps[i])

        # Load reference HDR
        ref_HDR_path = os.path.join(scene_path, self.ref_hdr_name)
        ref_HDR = get_image(ref_HDR_path, image_size=[h, w], is_crop=True)

        # Prepare U2Net inputs
        img1_ldr = in_LDRs[:, :, 0:c]
        img2_ldr = in_LDRs[:, :, c:c*2]
        img1_hdr = in_HDRs[:, :, 0:c]
        img2_hdr = in_HDRs[:, :, c:c*2]
        
        # Create summaries - concatenate LDR and HDR averages (6 channels total)
        sum_ldr = np.mean(in_LDRs.reshape(h, w, ns, c), axis=2)
        sum_hdr = np.mean(in_HDRs.reshape(h, w, ns, c), axis=2)
        sum1 = np.concatenate([sum_ldr, sum_hdr], axis=2)
        sum2 = np.concatenate([sum_ldr, sum_hdr], axis=2)

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