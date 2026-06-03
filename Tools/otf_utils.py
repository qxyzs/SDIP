# Tools/otf_utils.py
import torch
import torch.nn as nn

def fft2d(input_tensor, otf):
    f = torch.fft.fft2(input_tensor)
    fshift = torch.fft.fftshift(f)
    filtered = torch.mul(fshift, otf)
    ifshift = torch.fft.ifftshift(filtered)
    ifft = torch.fft.ifft2(ifshift)
    return ifft

def down_otf(img, otf):
    return torch.real(fft2d(img, otf))

class DownOTF(nn.Module):
    def __init__(self):
        super(DownOTF, self).__init__()
        self.pool_2X = nn.AvgPool2d(2, stride=2)

    def forward(self, x, y):
        return down_otf(x, y)