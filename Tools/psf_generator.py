# -*- coding: utf-8 -*-
import numpy as np
from scipy import special as spc

def get_AiryDisk(sizeImage, NA, wavelength, sizePixel, MA, CF=1.0, factor1=1.0, factor2=1.0):
    """
    Generate PSF (point spread function) and OTF (optical transfer function)
    for an objective lens with given parameters.
    
    Parameters:
        sizeImage: int, size of the output square array
        NA: float, numerical aperture
        wavelength: float, wavelength in nm
        sizePixel: float, pixel size in um
        MA: float, magnification
        CF: float, curvature factor
        factor1: float, power factor for OTF
        factor2: float, scaling factor for NA
    
    Returns:
        PSF0: 2D numpy array, normalized PSF
        OTF0: 2D numpy array, OTF (complex)
    """
    NA = NA * factor2
    wl = wavelength * 1e-9          # m
    k = 2 * np.pi / wl
    sizePixel = sizePixel * 1e-6    # m
    w = sizeImage
    wo = int(w / 2)

    x = np.linspace(0, w - 1, w)
    y = np.linspace(0, w - 1, w)
    aX, aY = np.meshgrid(x, y)

    aR = np.sqrt((aX - wo)**2 + (aY - wo)**2) * sizePixel
    aZ = (k * NA / MA) * aR

    temp_bj = spc.jv(1, aZ)
    PSF0 = (2 * temp_bj / (aZ + np.spacing(1)))**2
    PSF0[wo, wo] = 1
    PSF0 = PSF0 / np.sum(PSF0)

    PSF = np.fft.fftshift(PSF0)
    OTF = np.abs(np.fft.fft2(PSF))
    OTF0 = np.fft.fftshift(OTF)

    curMap = CF ** (aR / wo)
    OTF0 = curMap * OTF0
    OTF = np.fft.ifftshift(OTF0)

    PSF = np.abs(np.fft.ifft2(OTF))
    PSF0 = np.fft.ifftshift(PSF)
    OTF0 = OTF0 ** factor1

    return PSF0, OTF0