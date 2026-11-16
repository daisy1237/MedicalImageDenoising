
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
