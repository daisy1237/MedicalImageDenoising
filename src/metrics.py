
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

"""
   calculate the peak signal-to-noise ratio (psnr)
   args:
       original: reference image as a numpy array
       processed: image to evaluate, with the same shape as original
   returns:
       psnr value in decibels (higher values = better image quality)
"""
def calculatePsnr(original, processed):
    return peak_signal_noise_ratio(
        original,
        processed,
        data_range=255
    )

"""
    calculate the structural similarity index (ssim)
    args:
        original: reference image as a numpy array
        processed: image to evaluate, with the same shape as original
    returns:
        ssim value (closer to 1 = higher structural similarity)
"""
def calculateSsim(original, processed):
    return structural_similarity(
        original,
        processed,
        data_range=255,
        channel_axis=-1
    )

"""
    evaluate a processed image using psnr and ssim
    args:
        original: reference image as a numpy array
        processed: image obtained after noise addition or filtering
    returns:
        dictionary containing psnr and ssim values
"""
def evaluateImage(original, processed):
    return {
        "psnr": calculatePsnr(original, processed),
        "ssim": calculateSsim(original, processed)
    }
