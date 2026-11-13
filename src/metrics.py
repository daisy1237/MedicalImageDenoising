
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
def evaluateImage(original, processed,mask=None):
    if mask is not None:
        return {
            "psnr": calculateMaskedPsnr(original, processed, mask),
            "ssim": calculateMaskedSsim(original, processed, mask)
        }

    return {
        "psnr": calculatePsnr(original, processed),
        "ssim": calculateSsim(original, processed)
    }

"""
    calculate psnr using only pixels inside the retinal fov
    args:
        original: reference rgb image
        processed: evaluated rgb image
        mask: fov mask
    returns:
        psnr inside the retinal region
"""
def calculateMaskedPsnr(original, processed, mask):
    validPixels = mask > 0

    originalPixels = original[validPixels].astype(np.float64)
    processedPixels = processed[validPixels].astype(np.float64)

    mse = np.mean((originalPixels - processedPixels) ** 2)

    if mse == 0:
        return float("inf")

    return 10 * np.log10((255 ** 2) / mse)

"""
    calculate average ssim inside the retinal fov
    args:
        original: reference rgb image
        processed: evaluated rgb image
        mask: fov mask
    returns:
        mean ssim inside the retinal region
"""
def calculateMaskedSsim(original, processed, mask):
    _, ssimMap = structural_similarity(
        original,
        processed,
        data_range=255,
        channel_axis=-1,
        full=True
    )

    ssimMap = np.mean(ssimMap, axis=2)

    validPixels = mask > 0

    return float(np.mean(ssimMap[validPixels]))
