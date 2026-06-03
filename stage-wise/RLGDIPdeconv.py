# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='pywt')
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from Tools import np_to_torch
from Tools import compute_rl_prior, DownOTF, init_weights, get_AiryDisk
from Tools import process_image_with_background_removal
from unet import MediumUNet

def RLG_dip_deconv(img_np,
                   NA=1.3, wavelength=488, sizePixel=6.26, MA=100,
                   CF=1.0, factor1=1.0, factor2=1.0,
                   num_epochs=2000, inner_iterations=10,
                   data_weight=1.0, reg_weight=1.0, rl_weight=1.0,
                   learning_rate=1e-4,
                   rl_iterations=10, background_option=1,
                   verbose=True):
    """
    RL-guided DIP deconvolution with adjustable loss weights.
    Returns deconvolved image as 2D numpy array (range [0,1]).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if background_option not in [0, None]:
        img_np, _ = process_image_with_background_removal(
            img_np, background_option=background_option, constant=1.0, verbose=False
        )
    else:
        img_np = np.clip(img_np, 0, 1)

    if len(img_np.shape) == 2:
        img_np = img_np[np.newaxis, :, :]
    img_torch = np_to_torch(img_np).type(dtype)

    sizeImage = max(img_np.shape[1], img_np.shape[2])

    PSF0, OTF0 = get_AiryDisk(sizeImage, NA, wavelength, sizePixel, MA, CF, factor1, factor2)
    if PSF0.shape[0] != img_np.shape[1] or PSF0.shape[1] != img_np.shape[2]:
        import cv2
        PSF0 = cv2.resize(PSF0, (img_np.shape[2], img_np.shape[1]))
        PSF0 = PSF0 / np.sum(PSF0)
        psf_centered = np.fft.ifftshift(PSF0)
        otf = np.fft.fft2(psf_centered)
        OTF0 = np.fft.fftshift(otf)
    if not np.iscomplexobj(OTF0):
        OTF0 = OTF0.astype(np.complex64)
    OTF_tensor = torch.from_numpy(OTF0[None, None, :, :]).type(torch.complex64).to(device)

    rl_prior = compute_rl_prior(img_np, PSF0, rl_iterations=rl_iterations)
    rl_prior_tensor = np_to_torch(rl_prior).type(dtype).to(device)

    net = MediumUNet(n_channels=1, n_classes=1).to(device)
    init_weights(net, init_type='normal', init_gain=0.02)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    down_otf_module = DownOTF().to(device)
    z_current = img_torch.clone()
    mse = torch.nn.MSELoss().type(dtype)

    epoch_iterator = tqdm(range(num_epochs), desc='RLG-DIP Deconv', unit='epoch', disable=not verbose)
    for epoch in epoch_iterator:
        for _ in range(inner_iterations):
            optimizer.zero_grad()
            out = net(z_current)
            out_degraded = down_otf_module.forward(out, OTF_tensor)
            data_loss = mse(out_degraded, img_torch)
            reg_loss = mse(out, z_current)
            rl_loss = mse(out, rl_prior_tensor)
            total_loss = data_weight * data_loss + reg_weight * reg_loss + rl_weight * rl_loss
            total_loss.backward()
            optimizer.step()
        with torch.no_grad():
            z_current = out.detach().clone()
        epoch_iterator.set_postfix(loss=total_loss.item())

    deconvolved = z_current.cpu().numpy()[0, 0]
    return deconvolved