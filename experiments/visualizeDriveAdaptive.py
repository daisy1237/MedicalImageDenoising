
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.viewImage import loadImage, loadMask

from experiments.evaluateDriveAdaptive import (
    addNoise,
    gaussianFilter,
    adaptiveFilter,
    WEAK_KERNEL,
    WEAK_SIGMA,
    STRONG_KERNEL,
    STRONG_SIGMA,
    NOISE_SIGMA
)

from src.driveAlignment import (
    getOpticDiscCenter,
    alignDriveImage
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TEST_DIR = PROJECT_ROOT / "data" / "DRIVE" / "test"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "drive"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_IDS = ["04", "11", "15"]

BEST_ALPHA = 0.5
RANDOM_SEED = 42
CROP_SIZE = 120


def prepareImage(imageId, smoothedMap, validCount, rng):
    image = loadImage(
        TEST_DIR / "images" / f"{imageId}_test.tif"
    )

    fovMask = loadMask(
        TEST_DIR / "mask" / f"{imageId}_test_mask.gif"
    )

    vesselMask = loadMask(
        TEST_DIR / "rite_manual" / f"{imageId}_rite.png"
    )

    center = getOpticDiscCenter(
        image, fovMask, imageId, dataset="test"
    )

    alignedImage, alignedVessels, alignedFov = (
        alignDriveImage(
            image, vesselMask, fovMask, center
        )
    )

    clean = alignedImage.astype(np.float32) / 255.0

    noisy = addNoise(clean, NOISE_SIGMA, rng)

    weak = gaussianFilter(
        noisy, WEAK_KERNEL, WEAK_SIGMA
    )

    strong = gaussianFilter(
        noisy, STRONG_KERNEL, STRONG_SIGMA
    )

    validMask = (
        (alignedFov > 0) & (validCount > 0)
    )

    controlWeight = float(
        np.mean(smoothedMap[validMask])
    )

    control = (
        controlWeight * weak +
        (1 - controlWeight) * strong
    )

    tunedMap = np.clip(
        BEST_ALPHA * smoothedMap, 0, 1
    )

    adaptive = adaptiveFilter(
        weak, strong, tunedMap
    )

    return {
        "Original": clean,
        "Noisy": noisy,
        "Strong Gaussian": strong,
        "Noise-Aware Control": control,
        "Adaptive Smoothed Tuned": adaptive
    }, alignedVessels, alignedFov, tunedMap


def selectVesselCrop(vesselMask, fovMask):
    height, width = vesselMask.shape

    vessels = (vesselMask > 0).astype(np.uint8)
    fov = (fovMask > 0).astype(np.uint8)

    # Distance to the nearest background pixel.
    distance = cv2.distanceTransform(
        vessels,
        cv2.DIST_L2,
        5
    )

    # Use the skeleton so that thick vessel borders
    # are not incorrectly counted as thin vessels.
    skeleton = np.zeros_like(vessels)

    remaining = vessels.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3)
    )

    while np.any(remaining):
        eroded = cv2.erode(remaining, kernel)

        opened = cv2.dilate(eroded, kernel)

        edges = cv2.subtract(
            remaining, opened
        )

        skeleton = cv2.bitwise_or(
            skeleton, edges
        )

        remaining = eroded

    thinVessels = (
        (skeleton > 0) &
        (distance <= 2.0)
    ).astype(np.float32)

    # Estimate the optic-disc region from the aligned
    # image geometry. The target disc is near x=0.8*width.
    discX = int(width * 0.8)
    discY = height // 2

    yy, xx = np.ogrid[:height, :width]

    discDistance = np.sqrt(
        (xx - discX) ** 2 +
        (yy - discY) ** 2
    )

    thinVessels[discDistance < 100] = 0

    # Count thin-vessel skeleton pixels per candidate crop.
    density = cv2.boxFilter(
        thinVessels,
        ddepth=-1,
        ksize=(CROP_SIZE, CROP_SIZE),
        normalize=False
    )

    fovDensity = cv2.boxFilter(
        fov.astype(np.float32),
        ddepth=-1,
        ksize=(CROP_SIZE, CROP_SIZE),
        normalize=True
    )

    # Avoid regions crossing the FOV boundary.
    density[fovDensity < 0.98] = -1

    margin = CROP_SIZE // 2

    density[:margin, :] = -1
    density[-margin:, :] = -1
    density[:, :margin] = -1
    density[:, -margin:] = -1

    y, x = np.unravel_index(
        np.argmax(density),
        density.shape
    )

    if density[y, x] <= 0:
        raise ValueError(
            "No suitable thin-vessel crop found."
        )

    x1 = int(x - margin)
    y1 = int(y - margin)

    x2 = x1 + CROP_SIZE
    y2 = y1 + CROP_SIZE

    print(
        f"Selected thin-vessel crop: "
        f"x={x1}:{x2}, y={y1}:{y2}"
    )

    return x1, y1, x2, y2


def saveFullComparison(imageId, images):
    fig, axes = plt.subplots(
        1, len(images), figsize=(18, 5)
    )

    for ax, (name, image) in zip(axes, images.items()):
        ax.imshow(np.clip(image, 0, 1))
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(
        f"DRIVE Test {imageId} - Full Image Comparison"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / f"drive_{imageId}_full_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def saveCropComparison(imageId, images, crop):
    x1, y1, x2, y2 = crop

    fig, axes = plt.subplots(
        1, len(images), figsize=(18, 5)
    )

    for ax, (name, image) in zip(axes, images.items()):
        cropped = image[y1:y2, x1:x2]

        ax.imshow(np.clip(cropped, 0, 1))
        ax.set_title(name)
        ax.axis("off")

    fig.suptitle(
        f"DRIVE Test {imageId} - Vessel Detail"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / f"drive_{imageId}_vessel_detail.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def saveCropLocation(imageId, original, crop):
    x1, y1, x2, y2 = crop

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.imshow(original)

    rectangle = plt.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        fill=False,
        edgecolor="lime",
        linewidth=2
    )

    ax.add_patch(rectangle)

    ax.set_title(
        f"DRIVE Test {imageId} - Thin-Vessel Region"
    )

    ax.axis("off")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR /
        f"drive_{imageId}_crop_location.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

def saveProbabilityOverlay(imageId, original, probabilityMap):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    displayMax = float(np.max(probabilityMap))

    axes[0].imshow(original)
    axes[0].set_title("Original")

    heatmap = axes[1].imshow(
        probabilityMap,
        cmap="hot",
        vmin=0,
        vmax=displayMax
    )
    axes[1].set_title("Protection Weight Map")

    axes[2].imshow(original)
    axes[2].imshow(
        probabilityMap,
        cmap="hot",
        alpha=0.45,
        vmin=0,
        vmax=displayMax
    )
    axes[2].set_title("Anatomical Protection Overlay")

    for ax in axes:
        ax.axis("off")

    fig.colorbar(
        heatmap,
        ax=axes,
        fraction=0.025,
        pad=0.02
    )

    fig.savefig(
        FIGURES_DIR / f"drive_{imageId}_probability_overlay.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    smoothedMap = np.load(
        RESULTS_DIR / "drive_probability_smoothed.npy"
    )

    validCount = np.load(
        RESULTS_DIR / "drive_valid_count.npy"
    )

    # Reproduce the same noise realizations used in
    # evaluateDriveAdaptive.py, which processes all
    # 20 test images in numerical order.
    rng = np.random.default_rng(RANDOM_SEED)

    for number in range(1, 21):
        imageId = f"{number:02d}"

        images, vesselMask, fovMask, tunedMap = (
            prepareImage(
                imageId,
                smoothedMap,
                validCount,
                rng
            )
        )

        if imageId not in IMAGE_IDS:
            continue

        crop = selectVesselCrop(
            vesselMask, fovMask
        )

        saveCropLocation(
            imageId,
            images["Original"],
            crop
        )

        saveFullComparison(imageId, images)

        saveCropComparison(
            imageId, images, crop
        )

        saveProbabilityOverlay(
            imageId,
            images["Original"],
            tunedMap
        )

        print(f"Saved figures for test image {imageId}")

    print("\nAll figures saved in:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
