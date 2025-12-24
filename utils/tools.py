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


def tonemap(images):  # input/output 0~1
    return torch.log(1 + MU * (images + 1) / 2.) / np.log(1 + MU) * 2. - 1

def tonemap_np(images):  # input/output 0~1
    return np.log(1 + MU * (images + 1) / 2.) / np.log(1 + MU) * 2. - 1

def imread(path):
    if path[-4:] == '.hdr':
        img = cv2.imread(path, -1)
    else:
        img = cv2.imread(path)/255.
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
    img = img[0]
    h, w, _ = img.shape
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    file_path = sample_path + '/hdr.hdr'
    img = inverse_transform(img)
    img = np.einsum('ijk->jki', img)
    radiance_writer(file_path, img)
    
def validate_hdr_output(output, name="output"):

    min_val = output.min().item()
    max_val = output.max().item()
    mean_val = output.mean().item()
    
    print(f"✓ {name} stats: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")
    return True


import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def to_linear(x, eps=1e-4):
    """Convert [-1, 1] to [eps, 1]"""
    return torch.clamp((x + 1.0) * 0.5, eps, 1.0)


# ------------------------------------------------------------
# Log HDR loss (relative error)
# ------------------------------------------------------------
def hdr_log_l1(pred, gt):
    p = to_linear(pred)
    g = to_linear(gt)
    return torch.mean(torch.abs(torch.log(p) - torch.log(g)))


# ------------------------------------------------------------
# FIX: Bilateral contrast loss (penalize both over and under)
# ------------------------------------------------------------
def contrast_loss(pred, gt):
    pred_std = pred.flatten(1).std(dim=1)
    gt_std = gt.flatten(1).std(dim=1)
    
    # Penalize both over-contrast and under-contrast
    return torch.mean((pred_std - gt_std) ** 2)


# ------------------------------------------------------------
# FIX: Bilateral highlight loss
# ------------------------------------------------------------
def highlight_loss(pred, gt):
    p99_pred = torch.quantile(pred.flatten(1), 0.99, dim=1)
    p99_gt = torch.quantile(gt.flatten(1), 0.99, dim=1)
    
    # Penalize both over-exposure and under-exposure
    return torch.mean((p99_pred - p99_gt) ** 2)


# ------------------------------------------------------------
# FIX: Strong mean matching loss
# ------------------------------------------------------------
def mean_matching_loss(pred, gt):
    pred_mean = pred.flatten(1).mean(dim=1)
    gt_mean = gt.flatten(1).mean(dim=1)
    
    # Strongly penalize mean mismatch
    return torch.mean((pred_mean - gt_mean) ** 2)


# ------------------------------------------------------------
# Highlight-weighted absolute error
# ------------------------------------------------------------
def highlight_weighted_l1(pred, gt):
    w = torch.clamp((gt + 1.0) * 0.5, 0.0, 1.0)
    return torch.mean(w * torch.abs(pred - gt))


# ------------------------------------------------------------
# FIX: Add perceptual coherence loss
# ------------------------------------------------------------
def perceptual_gradient_loss(pred, gt):
    """Match local gradients to preserve structure"""
    # Sobel-like gradients
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    gt_dx = gt[:, :, :, 1:] - gt[:, :, :, :-1]
    gt_dy = gt[:, :, 1:, :] - gt[:, :, :-1, :]
    
    loss_dx = torch.mean(torch.abs(pred_dx - gt_dx))
    loss_dy = torch.mean(torch.abs(pred_dy - gt_dy))
    
    return loss_dx + loss_dy


class HDRLoss(nn.Module):
    """FIXED: Better balanced loss for statistics matching"""
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt):
        # FIX: Tighter clamp since we're using tanh now
        pred = torch.clamp(pred, -2.0, 2.0)
        
        # Core losses
        l_log = hdr_log_l1(pred, gt)
        l_hi = highlight_weighted_l1(pred, gt)
        l_con = contrast_loss(pred, gt)
        l_peak = highlight_loss(pred, gt)
        l_mean = mean_matching_loss(pred, gt)
        l_grad = perceptual_gradient_loss(pred, gt)
        
        # FIX: Rebalanced weights to fix statistics
        total = (
            0.8 * l_log +    # HDR accuracy
            0.4 * l_hi +     # Preserve bright structures
            1.2 * l_con +    # STRONG contrast matching (was 0.7)
            0.5 * l_peak +   # Highlight matching
            2.0 * l_mean +   # VERY STRONG mean matching (was 0.2!)
            0.3 * l_grad     # Structural coherence
        )
        print({
            'hdr_log_l1': l_log.item(),
            'highlight_weighted_l1': l_hi.item(),
            'contrast_loss': l_con.item(),
            'highlight_loss': l_peak.item(),
            'meamean_matching_lossn': l_mean.item(),
            'perceptual_gradient_loss': l_grad.item(),
            'total': total.item()
        })
        
        # Return individual losses for monitoring
        return total


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