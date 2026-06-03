import numpy as np
import torch
import tifffile

def load_16bit_tiff(filename):
    img = tifffile.imread(filename)
    img_float = img.astype(np.float32)
    if img.dtype == np.uint16:
        img_norm = img_float / 65535.0
    elif img.dtype == np.uint8:
        img_norm = img_float / 255.0
    else:
        img_norm = (img_float - img_float.min()) / (img_float.max() - img_float.min() + 1e-8)

    # Extract 2D image
    if len(img_norm.shape) == 4 and img_norm.shape[0] == 1 and img_norm.shape[1] == 1:
        img_2d = img_norm[0, 0]
    elif len(img_norm.shape) == 3 and img_norm.shape[0] == 1:
        img_2d = img_norm[0]
    elif len(img_norm.shape) == 4 and img_norm.shape[0] == 1:
        img_2d = img_norm[0, -1]
    elif len(img_norm.shape) == 3:
        img_2d = img_norm[0]
    elif len(img_norm.shape) == 2:
        img_2d = img_norm
    else:
        img_2d = img_norm.reshape(-1, img_norm.shape[-1])[:img_norm.shape[-2]]
    return img_2d

def save_16bit_tiff(image, filename, normalize=True):
    if torch.is_tensor(image):
        img_np = image.detach().cpu().numpy()
    else:
        img_np = image.copy()

    if len(img_np.shape) == 4:
        if img_np.shape[0] == 1:
            img_np = img_np[0]
            if img_np.shape[0] == 1:
                img_np = img_np[0]
            else:
                img_np = img_np.transpose(1, 2, 0)
        else:
            raise ValueError(f"unsupported batch size: {img_np.shape[0]}")
    elif len(img_np.shape) == 3:
        if img_np.shape[0] == 1 or img_np.shape[0] == 3:
            if img_np.shape[0] == 1:
                img_np = img_np[0]
            else:
                img_np = img_np.transpose(1, 2, 0)
    if len(img_np.shape) > 2 and img_np.shape[2] == 1:
        img_np = img_np[:, :, 0]

    if normalize:
        img_min = img_np.min()
        img_max = img_np.max()
        if img_max > img_min:
            img_np = (img_np - img_min) / (img_max - img_min)
        else:
            img_np = np.zeros_like(img_np)
        img_16bit = (img_np * 65535).astype(np.uint16)
    else:
        img_np = np.clip(img_np, 0, 1)
        img_16bit = (img_np * 65535).astype(np.uint16)

    tifffile.imwrite(filename, img_16bit)

def normalize_np(img):
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        return (img - img_min) / (img_max - img_min)
    else:
        return np.zeros_like(img)

def np_to_torch(img_np):
    """
    Convert numpy image (H,W) or (C,H,W) to torch tensor (1,C,H,W)
    """
    if len(img_np.shape) == 2:
        img_np = img_np[np.newaxis, :, :]          # (1, H, W)
    # Add batch dimension using unsqueeze(0)
    return torch.from_numpy(img_np).float().unsqueeze(0)