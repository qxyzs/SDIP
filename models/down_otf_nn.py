import numpy as np
import torch
import torch.nn as nn 
import cv2

def fft2d(input, otf):
    """
    修正的FFT频域滤波 - 保持原有函数名
    input: 输入图像 [B, C, H, W] 
    otf: 光学传递函数 [1, 1, H, W] 或 [H, W]
    """
    # 确保输入和OTF维度匹配
    if len(otf.shape) == 2:
        otf = otf.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # FFT变换到频域
    f = torch.fft.fft2(input)
    fshift = torch.fft.fftshift(f)
    
    # 频域滤波 - 复数乘法
    # OTF应该是复数，包含相位信息
    filtered = fshift * otf
    
    # 逆变换回空域
    ifshift = torch.fft.ifftshift(filtered)
    ifft = torch.fft.ifft2(ifshift)
    
    return torch.real(ifft)

def down_otf(img1, otf):
    """
    修正的OTF降质函数 - 保持原有函数名
    """
    ans1 = fft2d(img1, otf)
    return torch.real(ans1)

class down_otf_nn(nn.Module):
    def __init__(self):
        super(down_otf_nn, self).__init__()
        self.pool_2X = nn.AvgPool2d(2, stride=2)
    
    def forward(self, x, y):
        """
        x: 输入图像 [B, C, H, W]
        y: 光学传递函数 [1, 1, H, W] 或 [H, W]
        """
        # 确保OTF是复数张量
        if isinstance(y, np.ndarray):
            otf_tensor = torch.from_numpy(y).to(x.device)
        else:
            otf_tensor = y
            
        # 应用OTF降质
        degraded = down_otf(x, otf_tensor)
        
        # 如果需要降采样，取消注释下面这行
        # degraded = self.pool_2X(degraded)
        
        return degraded

# 新增：正确的OTF计算函数（您可以在主代码中使用）
def compute_otf_corrected(psf, target_shape=None):
    """
    正确计算OTF（保留相位信息）
    psf: 点扩散函数 [H, W]
    target_shape: 目标尺寸，如果为None则使用psf尺寸
    """
    if target_shape is not None and psf.shape != target_shape:
        psf = cv2.resize(psf, (target_shape[1], target_shape[0]))
    
    # 归一化PSF
    psf = psf.astype(np.float32)
    psf = psf / np.sum(psf)
    
    # 中心化PSF
    psf_centered = np.fft.ifftshift(psf)
    
    # 计算OTF - 不要取绝对值！
    otf = np.fft.fft2(psf_centered)
    
    # 中心化OTF
    otf_centered = np.fft.fftshift(otf)
    
    return otf_centered.astype(np.complex64)