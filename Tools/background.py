import numpy as np
import pywt

def background_estimation(imgs, th=1, dlevel=7, wavename='db6', iter=3):
    """
    Wavelet-based background estimation.
    
    Parameters:
        imgs: numpy.ndarray, shape (H, W) or (H, W, frames)
        th: int, 1 for thresholding, 0 otherwise
        dlevel: int, wavelet decomposition level
        wavename: str, wavelet name (e.g., 'db6')
        iter: int, number of iterations
    
    Returns:
        Background: numpy.ndarray, estimated background
    """
    if imgs.ndim == 2:
        imgs = imgs[:, :, np.newaxis]

    x, y, z = imgs.shape

    # Pad to square
    if x < y:
        pad_x = y - x
        pad_y = 0
    elif y < x:
        pad_x = 0
        pad_y = x - y
    else:
        pad_x = 0
        pad_y = 0

    if pad_x > 0 or pad_y > 0:
        imgs_padded = np.pad(imgs, ((0, pad_x), (0, pad_y), (0, 0)), mode='symmetric')
    else:
        imgs_padded = imgs.copy()

    padded_x, padded_y, _ = imgs_padded.shape
    Background = np.zeros((padded_x, padded_y, z), dtype=np.float32)

    for frame in range(z):
        initial = imgs_padded[:, :, frame].copy()
        res = initial.copy()

        for _ in range(iter):
            coeffs = pywt.wavedec2(res, wavename, level=dlevel)
            coeffs_new = [coeffs[0]]
            for i in range(1, len(coeffs)):
                coeffs_new.append((np.zeros_like(coeffs[i][0]),
                                   np.zeros_like(coeffs[i][1]),
                                   np.zeros_like(coeffs[i][2])))
            Biter = pywt.waverec2(coeffs_new, wavename)

            if th > 0:
                eps = np.sqrt(np.abs(res)) / 2
                ind = initial > (Biter + eps)
                res[ind] = Biter[ind] + eps[ind]

                coeffs = pywt.wavedec2(res, wavename, level=dlevel)
                coeffs_new = [coeffs[0]]
                for i in range(1, len(coeffs)):
                    coeffs_new.append((np.zeros_like(coeffs[i][0]),
                                       np.zeros_like(coeffs[i][1]),
                                       np.zeros_like(coeffs[i][2])))
                Biter = pywt.waverec2(coeffs_new, wavename)

        Biter = Biter[:padded_x, :padded_y]
        Background[:, :, frame] = Biter

    Background = Background[:x, :y, :]
    if z == 1:
        Background = Background[:, :, 0]
    return Background