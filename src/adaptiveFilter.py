
import numpy as np

from src.filters import bilateralFilter

"""
    apply adaptive denoising using a statistical probability map
    args:
        image: noisy 2D rgb image
        probabilityMap: statistical map with values in [0, 1]
        protectionStrength: maximum amount of original image preserved
    returns:
        adaptively denoised image
"""
def adaptiveFilter(image, probabilityMap, protectionStrength=0.7):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected a 2D RGB image")

    if probabilityMap.shape != image.shape[:2]:
        raise ValueError("probability map and image dimensions must match")

    if not 0 <= protectionStrength <= 1:
        raise ValueError("protection strength must be between 0 and 1")

    if not np.all(np.isfinite(probabilityMap)):
        raise ValueError("probability map contains invalid values")

    if np.any((probabilityMap < 0) | (probabilityMap > 1)):
        raise ValueError("probabilities must be between 0 and 1")

    filteredImage = bilateralFilter(image)

    weight = protectionStrength * probabilityMap
    weight = weight[:, :, np.newaxis]

    result = (
        weight * image.astype(np.float32)
        + (1 - weight) * filteredImage.astype(np.float32)
    )

    return np.clip(result, 0, 255).astype(np.uint8)


import cv2
import numpy as np


"""
    Gaussian filtering that ignores invalid pixels
"""
def maskedGaussian(image, support, kernelSize, sigma):
    valid = support.astype(np.float32)

    numerator = cv2.GaussianBlur(
        image.astype(np.float32) * valid,
        (kernelSize, kernelSize),
        sigmaX=sigma
    )

    denominator = cv2.GaussianBlur(
        valid,
        (kernelSize, kernelSize),
        sigmaX=sigma
    )

    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-6
    )

    result[~support] = 0

    return result


"""
    anatomical probability-guided adaptive filtering
"""
def adaptiveGaussianV2(image, probabilityMap, support):
    weak = maskedGaussian(
        image, support, 3, 0.6
    )

    strong = maskedGaussian(
        image, support, 7, 1.8
    )

    probability = np.clip(
        np.nan_to_num(probabilityMap, nan=0.0),
        0, 1
    )

    adaptive = (
        probability * weak
        + (1 - probability) * strong
    )

    adaptive[~support] = 0

    return weak, strong, adaptive


"""
    bilateral filtering that ignores invalid pixels
    using normalized weighted neighborhoods
"""
def maskedBilateral(image, support, diameter=7,
                    sigmaColor=0.1, sigmaSpace=3):
    image = image.astype(np.float32)
    valid = support.astype(bool)

    height, width = image.shape
    result = np.zeros_like(image)

    radius = diameter // 2

    paddedImage = np.pad(
        image, radius, mode="reflect"
    )
    paddedSupport = np.pad(
        valid, radius,
        mode="constant", constant_values=False
    )

    weightSum = np.zeros_like(image)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = paddedImage[
                radius + dy:radius + dy + height,
                radius + dx:radius + dx + width
            ]

            neighborValid = paddedSupport[
                radius + dy:radius + dy + height,
                radius + dx:radius + dx + width
            ]

            spatialWeight = np.exp(
                -(dx * dx + dy * dy)
                / (2 * sigmaSpace ** 2)
            )

            intensityWeight = np.exp(
                -((image - neighbor) ** 2)
                / (2 * sigmaColor ** 2)
            )

            weight = (
                spatialWeight
                * intensityWeight
                * neighborValid
            )

            result += weight * neighbor
            weightSum += weight

    np.divide(
        result, weightSum,
        out=result,
        where=weightSum > 1e-8
    )

    result[~valid] = 0

    return result


"""
    adaptive V3:
    bilateral in likely OPL regions,
    strong Gaussian elsewhere
"""
def adaptiveBilateralGaussian(image, probabilityMap, support):
    bilateral = maskedBilateral(
        image, support,
        diameter=7,
        sigmaColor=0.1,
        sigmaSpace=3
    )

    strong = maskedGaussian(
        image, support,
        kernelSize=7,
        sigma=1.8
    )

    probability = np.clip(
        np.nan_to_num(probabilityMap, nan=0.0),
        0, 1
    )

    adaptive = (
        probability * bilateral
        + (1 - probability) * strong
    )

    adaptive[~support] = 0

    return bilateral, strong, adaptive

"""
    noise-aware anatomical filtering
    the anatomical protection weight decrease as the estimated noise level increases
    """
def adaptiveGaussianV4(image, probabilityMap, support,
                       noiseSigma, k=15.0):

    weak = maskedGaussian(
        image, support,
        kernelSize=3,
        sigma=0.6
    )

    strong = maskedGaussian(
        image, support,
        kernelSize=7,
        sigma=1.8
    )

    probability = np.clip(
        np.nan_to_num(probabilityMap, nan=0.0),
        0.0, 1.0
    )

    noiseFactor = np.exp(-k * noiseSigma)

    weight = probability * noiseFactor

    adaptive = (
        weight * weak
        + (1.0 - weight) * strong
    )

    adaptive = adaptive.astype(np.float32)
    adaptive[~support] = 0

    return weak, strong, adaptive
