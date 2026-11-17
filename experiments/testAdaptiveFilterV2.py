
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2

from scipy.io import loadmat

from src.retinalNormalization import (
    normalizeRetinaTwoBoundaries
)


projRoot = Path(__file__).resolve().parent.parent
dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"


"""
    normalize image values to [0, 1]
"""
def normalizeImage(image):
    return image.astype(np.float32) / 255.0


"""
    add synthetic Gaussian noise
"""
def addGaussianNoise(image, sigma=0.08, seed=42):
    rng = np.random.default_rng(seed)

    noise = rng.normal(
        0, sigma, image.shape
    ).astype(np.float32)

    return np.clip(image + noise, 0, 1)


"""
    adaptive Gaussian filtering guided by
    the anatomical probability map
"""
def adaptiveGaussianFilter(image, probabilityMap):
    weakFiltered = cv2.GaussianBlur(
        image, (3, 3), sigmaX=0.6
    )

    strongFiltered = cv2.GaussianBlur(
        image, (7, 7), sigmaX=1.8
    )

    probability = np.nan_to_num(
        probabilityMap,
        nan=0.0
    )

    probability = np.clip(
        probability, 0, 1
    )

    adaptive = (
        probability * weakFiltered
        + (1 - probability) * strongFiltered
    )

    return (
        weakFiltered,
        strongFiltered,
        adaptive
    )


"""
    run one OCT experiment using the V2 map
"""
def testAdaptiveFilterV2(subjectId=9, sliceId=30):
    filePath = dataDir / f"Subject_{subjectId:02d}.mat"

    data = loadmat(filePath)

    image = data["images"][:, :, sliceId]
    layers = data["manualLayers1"][:, :, sliceId]

    normalizedImage, _, support = (
        normalizeRetinaTwoBoundaries(
            image, layers
        )
    )

    probabilityMap = np.load(
        resultsDir / "opl_probability_map_v2.npy"
    )

    validCount = np.load(
        resultsDir / "opl_valid_count_v2.npy"
    )

    if probabilityMap.shape != normalizedImage.shape:
        raise ValueError("Probability map shape mismatch")

    probability = np.where(
        validCount >= 5,
        np.nan_to_num(probabilityMap, nan=0.0),
        0.0
    ).astype(np.float32)

    clean = normalizeImage(normalizedImage)

    noisy = addGaussianNoise(clean)

    weak, strong, adaptive = adaptiveGaussianFilter(
        noisy, probability
    )

    # Display only the normalized anatomical region.
    images = [
        clean, noisy, weak, strong, adaptive
    ]

    titles = [
        "Reference OCT",
        "Noisy OCT",
        "Weak Gaussian",
        "Strong Gaussian",
        "Adaptive V2"
    ]

    fig, axes = plt.subplots(
        2, 3, figsize=(15, 9)
    )

    for ax, img, title in zip(
        axes.flat, images, titles
    ):
        displayed = img.copy()
        displayed[~support] = 0

        ax.imshow(
            displayed, cmap="gray",
            vmin=0, vmax=1
        )
        ax.set_title(title)
        ax.axis("off")

    axes.flat[5].imshow(
        probability, cmap="hot",
        vmin=0, vmax=1
    )
    axes.flat[5].set_title("OPL Probability V2")
    axes.flat[5].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)

    print("\n===== ADAPTIVE FILTER V2 =====")
    print(f"Subject: {subjectId:02d}")
    print(f"Slice: {sliceId}")
    print(f"Supported pixels: {np.sum(support)}")
    print(f"Probability max: {np.max(probability):.4f}")
