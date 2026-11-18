
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.viewImage import (
    loadImage,
    loadMask,
    imgDir,
    manualMasks,
    masks
)

from src.driveAlignment import (
    getOpticDiscCenter,
    alignDriveImage
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SMOOTH_SIGMA = 12


def buildDriveProbabilityMaps():
    imageFiles = sorted(imgDir.glob("*_training.tif"))

    vesselSum = None
    validCount = None

    for imagePath in imageFiles:
        imageId = imagePath.stem.split("_")[0]

        image = loadImage(imagePath)

        vesselMask = loadMask(
            manualMasks / f"{imageId}_manual1.gif"
        )

        fovMask = loadMask(
            masks / f"{imageId}_training_mask.gif"
        )

        discCenter = getOpticDiscCenter(
            image, fovMask, imageId
        )

        _, alignedVessels, alignedFov = alignDriveImage(
            image,
            vesselMask,
            fovMask,
            discCenter
        )

        vesselBinary = (alignedVessels > 0).astype(np.float32)
        validBinary = (alignedFov > 0).astype(np.float32)

        if vesselSum is None:
            vesselSum = np.zeros_like(vesselBinary)
            validCount = np.zeros_like(validBinary)

        vesselSum += vesselBinary * validBinary
        validCount += validBinary

        print(f"Processed image {imageId}")

    # Direct probability map
    directMap = vesselSum / np.maximum(validCount, 1)

    # Smooth both numerator and denominator to account
    # for anatomical variability and missing FOV support.
    smoothVesselSum = cv2.GaussianBlur(
        vesselSum,
        (0, 0),
        SMOOTH_SIGMA
    )

    smoothValidCount = cv2.GaussianBlur(
        validCount,
        (0, 0),
        SMOOTH_SIGMA
    )

    smoothedMap = smoothVesselSum / np.maximum(
        smoothValidCount, 1e-6
    )

    smoothedMap[validCount == 0] = 0

    directMap = np.clip(directMap, 0, 1)
    smoothedMap = np.clip(smoothedMap, 0, 1)

    np.save(
        RESULTS_DIR / "drive_probability_direct.npy",
        directMap
    )

    np.save(
        RESULTS_DIR / "drive_probability_smoothed.npy",
        smoothedMap
    )

    np.save(
        RESULTS_DIR / "drive_valid_count.npy",
        validCount
    )

    print("\n===== PROBABILITY MAP STATISTICS =====")

    validPixels = validCount > 0

    for name, probabilityMap in [
        ("Direct", directMap),
        ("Smoothed", smoothedMap)
    ]:
        values = probabilityMap[validPixels]

        print(f"\n{name} map:")
        print(f"  Min: {values.min():.4f}")
        print(f"  Max: {values.max():.4f}")
        print(f"  Mean: {values.mean():.4f}")
        print(f"  Median: {np.median(values):.4f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(directMap, cmap="hot", vmin=0, vmax=1)
    axes[0].set_title("Direct Vessel Probability")

    axes[1].imshow(smoothedMap, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Smoothed Vessel Probability")

    axes[2].imshow(validCount, cmap="viridis")
    axes[2].set_title("Valid Pixel Count")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
