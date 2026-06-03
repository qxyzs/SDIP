"""
Background Removal Tool - Multi-option Version
Depends on Tools.background_estimation and Tools.image_io
"""
import numpy as np
import matplotlib.pyplot as plt
from .background import background_estimation
from .image_io import load_16bit_tiff, save_16bit_tiff

def process_image_with_background_removal(img_array, background_option=1, constant=None, verbose=False):
    # Get max value
    if constant is None:
        constant = np.max(img_array)
    if constant == 0:
        constant = 1.0
    
    # Normalize to [0,1]
    img_norm = img_array.astype(np.float32) / constant
    
    # Preprocess and estimate background according to option
    if background_option == 1:
        backgrounds = background_estimation(img_norm / 2)
    elif background_option == 2:
        backgrounds = background_estimation(img_norm / 2.5)
    elif background_option == 3:
        medVal = np.mean(img_norm)
        sub_temp = img_norm.copy()
        sub_temp[sub_temp > medVal] = medVal
        backgrounds = background_estimation(sub_temp)
    elif background_option == 4:
        medVal = np.mean(img_norm) / 2
        sub_temp = img_norm.copy()
        sub_temp[sub_temp > medVal] = medVal
        backgrounds = background_estimation(sub_temp)
    elif background_option == 5:
        medVal = np.mean(img_norm) / 2.5
        sub_temp = img_norm.copy()
        sub_temp[sub_temp > medVal] = medVal
        backgrounds = background_estimation(sub_temp)
    elif background_option == 6:
        backgrounds = np.zeros_like(img_norm)
    else:
        backgrounds = background_estimation(img_norm / 2)
    
    # Subtract background
    result = img_norm - backgrounds
    result[result < 0] = 0
    
    # Re-normalize to [0,1]
    if result.max() > 0:
        result_norm = result / result.max()
    else:
        result_norm = result
    
    background_norm = backgrounds
    
    return result_norm, background_norm

def apply_background_removal_to_file(input_file_path, output_dir, background_option=1):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    img_np = load_16bit_tiff(input_file_path)
    import tifffile
    original = tifffile.imread(input_file_path)
    constant = np.max(original)
    
    result_norm, bg_norm = process_image_with_background_removal(
        img_np * constant,
        background_option=background_option,
        constant=constant,
        verbose=False
    )
    
    bg_16bit = (bg_norm * 65535).astype(np.uint16)
    bg_path = os.path.join(output_dir, os.path.basename(input_file_path).replace('.tiff', '_background_16bit.tiff'))
    save_16bit_tiff(bg_16bit, bg_path, normalize=False)
    
    res_16bit = (result_norm * 65535).astype(np.uint16)
    res_path = os.path.join(output_dir, os.path.basename(input_file_path).replace('.tiff', '_no_background_16bit.tiff'))
    save_16bit_tiff(res_16bit, res_path, normalize=False)
    
    return result_norm, bg_norm