
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

projRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projRoot))

from src.noise import addNoise
from src.filters import bilateralFilter
from src.adaptiveFilter import adaptiveFilter
from src.metrics import evaluateImage


trainingDir = projRoot / "data" / "DRIVE" / "training"
imagesDir = trainingDir / "images"
fovMasksDir = trainingDir / "mask"
resultsDir = projRoot / "results"


"""
    compare bilateral and probability-based adaptive denoising
    on a retinal image
"""
def runAdaptiveTest(imageId="21", sigma=20, protectionStrength=0.7):
    resultsDir.mkdir(parents=True, exist_ok=True)

    imagePath = imagesDir / f"{imageId}_training.tif"
    fovPath = fovMasksDir / f"{imageId}_training_mask.gif"
    probabilityMapPath = resultsDir / "vessel_probability_map.npy"

    original = cv2.imread(str(imagePath))

    if original is None:
        raise FileNotFoundError(f"Could not load image: {imagePath}")

    original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

    if not fovPath.exists():
        raise FileNotFoundError(f"Missing FOV mask: {fovPath}")

    fovMask = np.array(Image.open(fovPath).convert("L"))

    if not probabilityMapPath.exists():
        raise FileNotFoundError(
            f"Missing probability map: {probabilityMapPath}"
        )

    probabilityMap = np.load(probabilityMapPath)

    if probabilityMap.shape != original.shape[:2]:
        raise ValueError(
            "Probability map and image dimensions do not match"
        )

    # add controlled gaussian noise
    noisyImage = addNoise(original, sigma=sigma, seed=42)

    # apply reference and proposed filters
    bilateralImage = bilateralFilter(noisyImage)

    adaptiveImage = adaptiveFilter(
        noisyImage,
        probabilityMap,
        protectionStrength=protectionStrength
    )

    # evaluate image quality inside the retinal FOV
    images = {
        "noisy": noisyImage,
        "bilateral": bilateralImage,
        "adaptive": adaptiveImage
    }

    print("\n===== ADAPTIVE FILTER TEST =====")
    print(f"image: {imageId}")
    print(f"noise sigma: {sigma}")
    print(f"protection strength: {protectionStrength}")

    print(f"\n{'method':<20} {'psnr (dB)':<15} {'ssim':<10}")
    print("-" * 47)

    for method, processed in images.items():
        metrics = evaluateImage(original, processed, fovMask)

        print(
            f"{method:<20} "
            f"{metrics['psnr']:<15.4f} "
            f"{metrics['ssim']:<10.4f}"
        )

    # visualize results
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    displayImages = [
        (original, "original image"),
        (noisyImage, "noisy image"),
        (bilateralImage, "bilateral filter"),
        (probabilityMap, "probability map"),
        (adaptiveImage, "adaptive filter"),
        (
            cv2.absdiff(bilateralImage, adaptiveImage),
            "difference: bilateral vs adaptive"
        )
    ]

    for ax, (image, title) in zip(axes.flat, displayImages):
        if image.ndim == 2:
            ax.imshow(image, cmap="hot", vmin=0, vmax=1)
        else:
            ax.imshow(image)

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()

    outputPath = resultsDir / f"adaptive_test_{imageId}.png"
    plt.savefig(outputPath, dpi=300)

    plt.show()

    print(f"\nvisualization saved to: {outputPath}")
