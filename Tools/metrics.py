import numpy as np
from skimage.metrics import structural_similarity as ssim

def compare_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_val = max(img1.max(), img2.max())
    psnr = 10 * np.log10(max_val ** 2 / mse)
    return psnr

def compare_ssim(img1, img2, data_range=1.0):
    return ssim(img1, img2, data_range=data_range)