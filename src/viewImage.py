from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from src import filters
from src import noise
from src.metrics import evaluateImage

projRoot = Path(__file__).resolve().parent.parent
trainingDir = projRoot / "data" / "DRIVE" / "training"

imgDir = trainingDir / "images"
manualMasks = trainingDir / "1st_manual"
masks = trainingDir / "mask"

def loadImage(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def loadMask(path):
    return np.array(Image.open(path).convert("L"))

def visualizeSample(imageId):
    imagePath = imgDir / f"{imageId}_training.tif"
    vesselPath = manualMasks / f"{imageId}_manual1.gif"
    fovPath = masks / f"{imageId}_training_mask.gif"

    image = loadImage(imagePath)
    vesselMask = loadMask(vesselPath)
    fovMask = loadMask(fovPath)

    # add controlled noise
    noisyImage = noise.addNoise(image, sigma=20)

    # apply all reference filters to the same noisy image
    gaussianImage = filters.gaussianFilter(noisyImage)
    bilateralImage = filters.bilateralFilter(noisyImage)
    nlmImage = filters.nonLocalMeansFilter(noisyImage)

    # evaluate noisy and filtered images
    results = {
        "noisy image": evaluateImage(image, noisyImage),
        "gaussian": evaluateImage(image, gaussianImage),
        "bilateral": evaluateImage(image, bilateralImage),
        "non local means": evaluateImage(image, nlmImage)
    }

    # display numerical results
    print("\n===== IMAGE QUALITY EVALUATION =====")
    print(f"{'method':<20} {'psnr (dB)':<15} {'ssim':<10}")
    print("-" * 47)

    for method, metrics in results.items():
        print(
            f"{method:<20} "
            f"{metrics['psnr']:<15.4f} "
            f"{metrics['ssim']:<10.4f}"
        )

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    images = [
        (image, "original image"),
        (noisyImage, "gaussian noise (sigma=20)"),
        (gaussianImage, "gaussian filter"),
        (bilateralImage, "bilateral filter"),
        (nlmImage, "non local means"),
        (vesselMask, "vessel segmentation")
    ]

    for ax, (img, title) in zip(axes.flat, images):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)

        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    plt.show()


