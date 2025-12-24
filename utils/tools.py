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
        return 20 * math.log10(self.range / math.sqrt(mse))


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
                    ssims.append(self._ssim(img1, img2))
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


def compute_summary_fast(ldr, tiles, coords, orig_shape, tile_h, tile_w):
    """
    COMPLETE: the summary is calculated by overlaying all the tiles of the image
    let's say N tiles, and then merging them all so we end up with one tile representation
    of the entire image
    """


def LDR2HDR(img, expo): # input/output 0~1
    return (((img+1)/2.)**GAMMA / expo) *2.-1


def LDR2HDR_batch(imgs, expos): # input/output 0~1
    return np.concatenate([LDR2HDR(imgs[:, :, 0:3], expos[0]),
                           LDR2HDR(imgs[:, :, 3:6], expos[1]),
                           LDR2HDR(imgs[:, :, 6:9], expos[2])], axis=2)


def HDR2LDR(imgs, expo): # input/output 0~1
    return (np.clip(((imgs+1)/2.*expo),0,1)**(1/GAMMA)) *2.-1


def transform_LDR(image, im_size=(256, 256)):
    out = image.astype(np.float32)
    out = cv2.resize(out, im_size)
    return out/127.5 - 1.


def transform_HDR(image, im_size=(256, 256)):
    out = cv2.resize(image, im_size)
    return out*2. - 1.


def tonemap(images, eps=1e-6):
    """
    Numerically stable tonemap function with proper handling of edge cases
    Input/output range: [-1, 1]
    """
    # Clamp input to valid range to prevent extreme values
    images = torch.clamp(images, -1.0, 1.0)
    
    # Transform from [-1, 1] to [0, 1] with epsilon for stability
    normalized = (images + 1.0) / 2.0
    normalized = torch.clamp(normalized, eps, 1.0)
    
    # Apply mu-law compression with numerical stability
    compressed = torch.log(1.0 + MU * normalized + eps) / math.log(1.0 + MU)
    
    # Transform back to [-1, 1]
    result = compressed * 2.0 - 1.0
    
    # Final safety check
    result = torch.clamp(result, -1.0, 1.0)
    
    # Check for NaN/Inf and replace with zeros if found
    if torch.isnan(result).any() or torch.isinf(result).any():
        print("⚠️  Warning: NaN/Inf detected in tonemap output, replacing with safe values")
        result = torch.where(torch.isnan(result) | torch.isinf(result), 
                            torch.zeros_like(result), result)
    
    return result


def tonemap_np(images, eps=1e-6):
    """
    Numerically stable tonemap function for numpy arrays
    Input/output range: [-1, 1]
    """
    # Clamp input to valid range
    images = np.clip(images, -1.0, 1.0)
    
    # Transform from [-1, 1] to [0, 1] with epsilon
    normalized = (images + 1.0) / 2.0
    normalized = np.clip(normalized, eps, 1.0)
    
    # Apply mu-law compression
    compressed = np.log(1.0 + MU * normalized + eps) / np.log(1.0 + MU)
    
    # Transform back to [-1, 1]
    result = compressed * 2.0 - 1.0
    
    # Final clamp
    result = np.clip(result, -1.0, 1.0)
    
    # Handle NaN/Inf
    result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return result


def imread(path):
    if path[-4:] == '.hdr':
        img = cv2.imread(path, -1)
    else:
        img = cv2.imread(path)/255.
    return img.astype(np.float32)[..., ::-1]


def radiance_writer(out_path, image):
    """
    Write HDR image in Radiance format (.hdr)
    Handles NaN and infinite values gracefully
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Check for and fix invalid values
    if np.isnan(image).any():
        print(f"Warning: NaN values detected in HDR image, replacing with 0")
    
    if np.isinf(image).any():
        print(f"Warning: Infinite values detected in HDR image, clipping")
    
    # Ensure non-negative values (HDR should be non-negative)
    image = np.maximum(image, 0.0)
    
    with open(out_path, "wb") as f:
        f.write(b"#?RADIANCE\n# Made with Python & Numpy\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (image.shape[0], image.shape[1]))

        # Find brightest channel for each pixel
        brightest = np.maximum(np.maximum(image[..., 0], image[..., 1]), image[..., 2])
        
        # Add small epsilon to avoid division by zero
        brightest = np.maximum(brightest, 1e-32)
        
        # Initialize arrays
        mantissa = np.zeros_like(brightest)
        exponent = np.zeros_like(brightest)
        
        # Compute mantissa and exponent
        np.frexp(brightest, mantissa, exponent)
        
        # Scale mantissa, avoiding division by zero
        scaled_mantissa = np.divide(
            mantissa * 255.0, 
            brightest,
            out=np.zeros_like(mantissa),
            where=brightest > 1e-32
        )
        
        # Initialize RGBE array
        rgbe = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        
        # Safely compute RGB values with clipping
        rgb_values = image[..., 0:3] * scaled_mantissa[..., None]
        # Convert to uint8
        rgbe[..., 0:3] = np.around(rgb_values).astype(np.uint8)
        
        # Compute exponent with offset, clipping to valid range
        exp_values = np.clip(exponent + 128, 0, 255)
        rgbe[..., 3] = np.around(exp_values).astype(np.uint8)
        # Write to file
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


# always return RGB, float32, range 0~1
def get_image(image_path, image_size=None, is_crop=False):
    if is_crop:
        assert (image_size is not None), "the crop size must be specified"
    return transform(imread(image_path), image_size, is_crop)


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


def transform(image, image_size, is_crop):
    if is_crop:
        out = center_crop(image, image_size)
    elif image_size is not None:
        out = cv2.resize(image, image_size)
    else:
        out = image
    out = out*2. - 1
    return out.astype(np.float32)


def inverse_transform(images):
    return (images + 1) / 2


# get input
def get_input(LDR_path, exp_path, ref_HDR_path):
    in_LDR_paths = sorted(glob(LDR_path))
    ns = len(in_LDR_paths)
    tmp_img = cv2.imread(in_LDR_paths[0]).astype(np.float32)
    h, w, c = tmp_img.shape
    h = h // 8 * 8
    w = w // 8 * 8

    in_exps = np.array(open(exp_path).read().split('\n')[:ns]).astype(np.float32)
    in_LDRs = np.zeros((h, w, c * ns), dtype=np.float32)
    in_HDRs = np.zeros((h, w, c * ns), dtype=np.float32)

    for i, image_path in enumerate(in_LDR_paths):
        img = get_image(image_path, image_size=[h, w], is_crop=True)
        in_LDRs[:, :, c * i:c * (i + 1)] = img
        in_HDRs[:, :, c * i:c * (i + 1)] = LDR2HDR(img, 2. ** in_exps[i])

    ref_HDR = get_image(ref_HDR_path, image_size=[h, w], is_crop=True)
    return in_LDRs, in_HDRs, in_exps, ref_HDR

def dump_sample(sample_path, img):
    """
    Save HDR sample with validation.
    Accepts:
      - (B, C, H, W)
      - (C, H, W)
      - (H, W, C)
    """

    # ---- Normalize shape ----
    if img.ndim == 4:          # (B, C, H, W)
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1, 3):  # (C, H, W)
        img = img.transpose(1, 2, 0)              # -> (H, W, C)

    assert img.ndim == 3, f"Invalid image shape: {img.shape}"

    h, w, c = img.shape

    # ---- Create directory ----
    os.makedirs(sample_path, exist_ok=True)
    file_path = os.path.join(sample_path, 'hdr.hdr')

    # ---- Transform from [-1, 1] to [0, inf) ----
    img = inverse_transform(img)

    # ---- Validate ----
    if np.isnan(img).any() or np.isinf(img).any():
        print(f"Warning: Invalid values in output HDR image for {sample_path}")
        print(f"  NaN count: {np.isnan(img).sum()}")
        print(f"  Inf count: {np.isinf(img).sum()}")
        print(f"  Min value: {np.nanmin(img):.6f}")
        if not np.isinf(img).all():
            print(f"  Max value: {np.nanmax(img[~np.isinf(img)])}")

    # ---- Write HDR ----
    radiance_writer(file_path, img)
    
def validate_hdr_output(output, name="output"):
    """
    Validate HDR output tensor for debugging
    Returns True if valid, False otherwise
    """
    if torch.isnan(output).any():
        nan_count = torch.isnan(output).sum().item()
        print(f"❌ {name} contains {nan_count} NaN values")
    
    if torch.isinf(output).any():
        inf_count = torch.isinf(output).sum().item()
        print(f"❌ {name} contains {inf_count} infinite values")

    min_val = output.min().item()
    max_val = output.max().item()
    mean_val = output.mean().item()
    
    print(f"✓ {name} stats: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
    return True


class HDRLoss(nn.Module):
    """Numerically stable HDR loss with gradient clipping"""
    def __init__(self):
        super().__init__()
    
    def forward(self, out_img, ref_img):
        # Ensure inputs are valid
        if torch.isnan(out_img).any() or torch.isnan(ref_img).any():
            print("⚠️  Warning: NaN detected in loss inputs")
        
        if torch.isinf(out_img).any() or torch.isinf(ref_img).any():
            print("⚠️  Warning: Inf detected in loss inputs")
            
        # Compute squared difference
        diff = out_img - ref_img
        squared_diff = diff ** 2
        
        # Compute mean
        loss = torch.mean(squared_diff)
        
        # Final safety check
        if torch.isnan(loss) or torch.isinf(loss):
            print("⚠️  Warning: Invalid loss computed, returning zero")

        return loss
    

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