# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from Tools import np_to_torch
from Tools import init_weights
from unet import UNet

def aSeqDIP_denoise(img_np,
                    num_epochs=2000,
                    inner_iterations=10,
                    learning_rate=1e-4,
                    data_weight=1.0,
                    reg_weight=1.0,
                    has_gt=False,
                    gt_np=None,
                    verbose=True):
    """
    DIP denoising with adjustable data and regularization weights.
    Returns denoised image as 2D numpy array (range [0,1]).
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    if len(img_np.shape) == 2:
        img_np = img_np[np.newaxis, :, :]
    img_torch = np_to_torch(img_np).type(dtype)

    net = UNet(n_channels=1, n_classes=1).to(device)
    init_weights(net, init_type='normal', init_gain=0.02)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    net_input = img_torch.clone()
    mse = torch.nn.MSELoss().type(dtype)

    best_combined_score = -1
    best_combined_img = None
    output_at_x = None

    if has_gt and gt_np is not None:
        if len(gt_np.shape) == 2:
            gt_np = gt_np[np.newaxis, :, :]
        gt_torch = np_to_torch(gt_np).type(dtype)
        from Tools import compare_psnr, compare_ssim

    epoch_iterator = tqdm(range(num_epochs), desc='aSeqDIP Denoising', unit='epoch', disable=not verbose)
    for epoch in epoch_iterator:
        for _ in range(inner_iterations):
            optimizer.zero_grad()
            out = net(net_input)
            loss_data = mse(out, img_torch)
            loss_reg = mse(net_input, out)
            total_loss = data_weight * loss_data + reg_weight * loss_reg
            total_loss.backward()
            optimizer.step()
        with torch.no_grad():
            net_input = out.detach().clone()

        if epoch == 1000:
            output_at_x = out.detach().cpu().numpy()[0, 0].copy()

        if has_gt and (epoch % 10 == 0 or epoch == num_epochs - 1):
            current_out_np = out.detach().cpu().numpy()[0, 0]
            current_gt_np = gt_torch.cpu().numpy()[0, 0]
            psnr_val = compare_psnr(current_gt_np, current_out_np)
            ssim_val = compare_ssim(current_gt_np, current_out_np)
            combined = (psnr_val / 50) + ssim_val
            if combined > best_combined_score:
                best_combined_score = combined
                best_combined_img = current_out_np.copy()
                epoch_iterator.set_postfix(loss=total_loss.item())
    if has_gt and best_combined_img is not None:
        result_img = best_combined_img
    else:
        if output_at_x is None:
            output_at_x = net_input.cpu().numpy()[0, 0]
        result_img = output_at_x
    return result_img