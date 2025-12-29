import time
import torch
import numpy as np
from math import exp
import torch.nn.functional as F
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import math
import torch.nn as nn
import cv2
import os
from glob import glob
import pickle


MU = 5000.0  # Mu-law compression parameter
GAMMA = 2.2  # Gamma for LDR/HDR conversion

class PSNR():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * np.log10(self.range / np.sqrt(mse))

class SSIM():
    def __init__(self, range=1):
        self.range = range

    def __call__(self, img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(self._ssim(img1[:, :, i], img2[:, :, i]))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    def _ssim(self, img1, img2):
        C1 = (0.01 * self.range) ** 2
        C2 = (0.03 * self.range) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()
    
    
def gettime():
    current_time = time.localtime()
    time_str = str(current_time.tm_year) + '-' + str(current_time.tm_mon) + '-' + str(current_time.tm_mday) + \
               '-' + str(current_time.tm_hour)
    return time_str


def LDR2HDR(img, expo):
    """Convert LDR to HDR with exposure correction"""
    # img_01 = np.clip((img + 1.0) / 2.0, 0.0, 1.0)
    img_hdr = np.power(img, GAMMA)
    img_hdr = img_hdr / expo  # MULTIPLY not divide!
    img_compressed = np.log(1.0 + MU * img_hdr) / np.log(1.0 + MU)
    img_normalized = img_compressed * 2.0 - 1.0
    print()
    return img_normalized.astype(np.float32)

from pathlib import Path

def imread(path):
    if isinstance(path, Path):
        path = str(path)
    if path.lower().endswith('.hdr'):
        img = cv2.imread(path, -1)
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            img = img.astype(np.float32) / 65535.0
    return img.astype(np.float32)[..., ::-1]


def radiance_writer(out_path, image):
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" %(image.shape[0], image.shape[1]))

        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        np.frexp(brightest, mantissa, exponent)
        scaled_mantissa = mantissa * 255.0 / brightest
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        rgbe[..., 0:3] = np.around(image[..., 0:3] * scaled_mantissa[..., None])
        rgbe[..., 3] = np.around(exponent + 128)

        rgbe.flatten().tofile(f)

def store_patch(h1, h2, w1, w2, in_LDRs, in_exps, ref_HDR, ref_LDRs, ref_exps, save_path, save_id):
    res = {
        'in_LDR': in_LDRs,
        'ref_LDR': ref_LDRs,
        'ref_HDR': ref_HDR,
        'in_exp': in_exps,
        'ref_exp': ref_exps,
    }
    with open(save_path + '/' + str(save_id) + '.pkl', 'wb') as pkl_file:
        pickle.dump(res, pkl_file)


def get_patch_from_file(pkl_path, pkl_id):
    with open(pkl_path + '/' + str(pkl_id) + '.pkl', 'rb') as pkl_file:
        res = pickle.load(pkl_file)
    return res


def get_image(image_path, image_size=None, is_crop=False):
    if is_crop:
        assert image_size is not None, "the crop size must be specified"

    # ---- normalize path ----
    if isinstance(image_path, Path):
        image_path = str(image_path)

    img = imread(image_path)

    is_hdr = image_path.lower().endswith(('.hdr', '.exr'))

    return transform(
        img,
        image_size,
        is_crop,
        is_hdr=is_hdr
    )

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
    return img


def center_crop(x, image_size):
    crop_h, crop_w = image_size
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return cv2.resize(x[max(0, j):min(h, j+crop_h), max(0, i):min(w, i+crop_w)], (crop_w, crop_h))


def transform(image, image_size, is_crop, is_hdr=False):
    if is_crop:
        out = center_crop(image, image_size)
    elif image_size is not None:
        out = cv2.resize(image, image_size)
    else:
        out = image
    
    if is_hdr:
        # HDR: Apply log compression
        out_compressed = np.log(1.0 + MU * out) / np.log(1.0 + MU)
        out = out_compressed * 2.0 - 1.0
    
    return out.astype(np.float32)

def inverse_transform(images, MU=5000.0):
    # images: tensor with expected range [-1,1]
    compressed = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)  # <- clamp to [0,1]
    hdr_radiance = (torch.pow(1.0 + MU, compressed) - 1.0) / MU
    return hdr_radiance

def inverse_transform_np(images, MU=5000.0):
    compressed = np.clip((images + 1.0) / 2.0, 0.0, 1.0)  # <- clip
    hdr_radiance = (np.power(1.0 + MU, compressed) - 1.0) / MU
    return hdr_radiance


def dump_sample(sample_path, img):
    img = img[0]
    h, w, _ = img.shape
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    file_path = sample_path + '/hdr.hdr'
    img = inverse_transform_np(img)
    img = np.einsum('ijk->jki', img)
    radiance_writer(file_path, img)
    


class HDRLoss(nn.Module):
    """
    Multi-component loss for HDR reconstruction that addresses:
    - Range control (prevents overshooting)
    - Texture preservation (reduces graininess)
    - Perceptual quality
    """
    def __init__(self, 
                 mse_weight=1.0,
                 l1_weight=1.0,
                 gradient_weight=1.0,
                 range_penalty_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.l1_weight = l1_weight
        self.gradient_weight = gradient_weight
        self.range_penalty_weight = range_penalty_weight

    def gradient_loss(self, pred, target):
        """
        Gradient loss - helps preserve edges and reduces smoothing artifacts
        """
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                                dtype=pred.dtype, device=pred.device).view(1, 1, 3, 3)
        
        # Expand for all channels
        sobel_x = sobel_x.repeat(pred.size(1), 1, 1, 1)
        sobel_y = sobel_y.repeat(pred.size(1), 1, 1, 1)
        
        # Compute gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1, groups=pred.size(1))
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1, groups=pred.size(1))
        target_grad_x = F.conv2d(target, sobel_x, padding=1, groups=target.size(1))
        target_grad_y = F.conv2d(target, sobel_y, padding=1, groups=target.size(1))
        
        # L1 loss on gradients
        grad_loss = F.l1_loss(pred_grad_x, target_grad_x) + \
                    F.l1_loss(pred_grad_y, target_grad_y)
        
        return grad_loss
    
    def range_penalty(self, pred, target):
        """
        Penalize predictions that exceed target range - prevents overshooting
        """
        target_min = target.min()
        target_max = target.max()
        
        # Penalize values outside target range
        below_min = F.relu(target_min - pred)
        above_max = F.relu(pred - target_max)
        
        penalty = below_min.mean() + above_max.mean()
        
        # Extra penalty for extreme overshoots (>1.0 when target max < 1.0)
        if target_max < 1.0:
            extreme_overshoot = F.relu(pred - 1.0)
            penalty += extreme_overshoot.mean() * 2.0
        
        return penalty

    def forward(self, pred, target):
        """
        Combined loss function
        """
        if torch.isnan(pred).any():
            print(f"⚠️  NaN in pred gradients")
            
        if torch.isnan(target).any():
            print(f"⚠️  NaN in target gradients")
        
        # Basic reconstruction losses
        mse_loss = F.mse_loss(pred, target)
        l1_loss = F.l1_loss(pred, target)
        
        # Perceptual quality losses
        grad_loss = self.gradient_loss(pred, target)
        
        # Range control
        range_penalty_val = self.range_penalty(pred, target)
        
        # Combine all losses
        total_loss = (
            self.mse_weight * mse_loss +
            self.l1_weight * l1_loss +
            self.gradient_weight * grad_loss +
            self.range_penalty_weight * range_penalty_val
        )
        
        # Log individual components for monitoring
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'l1': l1_loss.item(),
            'gradient': grad_loss.item(),
            'range_penalty': range_penalty_val.item()
        }
        
        return total_loss, loss_dict
    
    
from torch.optim.lr_scheduler import LambdaLR


class LambdaStepLR(LambdaLR):
  def __init__(self, optimizer, lr_lambda, last_step=-1):
    super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

  @property
  def last_step(self):
    return self.last_epoch

  @last_step.setter
  def last_step(self, v):
    self.last_epoch = v


class PolyLR(LambdaStepLR):
  def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
    super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)


class SquaredLR(LambdaStepLR):
  def __init__(self, optimizer, max_iter, last_step=-1):
    super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**2, last_step)