import numpy as np
from scipy.signal import fftconvolve

def richardson_lucy_deconvolution(image, psf, num_iter=20, boundary_mode='reflect'):
    image = image.astype(np.float64)
    psf = psf.astype(np.float64)
    psf_norm = psf / np.sum(psf)
    psf_flipped = np.flip(psf_norm)

    pad_width = max(psf_norm.shape[0], psf_norm.shape[1], 32)
    image_padded = np.pad(image, pad_width, mode=boundary_mode)
    estimate = image_padded.copy()

    for i in range(num_iter):
        blurred = fftconvolve(estimate, psf_norm, mode='same')
        blurred[blurred == 0] = 1e-12
        ratio = image_padded / blurred
        correction = fftconvolve(ratio, psf_flipped, mode='same')
        estimate = estimate * correction

    deconvolved = estimate[pad_width:-pad_width, pad_width:-pad_width]
    deconvolved = np.maximum(deconvolved, 0)
    deconvolved = (deconvolved - deconvolved.min()) / (deconvolved.max() - deconvolved.min() + 1e-8)
    return deconvolved

def compute_rl_prior(img_noisy_np, psf, rl_iterations=30):
    if len(img_noisy_np.shape) == 3 and img_noisy_np.shape[0] == 1:
        img_2d = img_noisy_np[0]
    else:
        img_2d = img_noisy_np

    if len(psf.shape) > 2:
        psf_2d = psf[0] if psf.shape[0] == 1 else psf
    else:
        psf_2d = psf

    rl_result = richardson_lucy_deconvolution(img_2d, psf_2d, num_iter=rl_iterations)
    return rl_result[np.newaxis, :, :]