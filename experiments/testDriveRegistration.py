
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


def testDriveRegistration():
    imageFiles = sorted(imgDir.glob("*_training.tif"))

    alignedImages = []
    alignedVessels = []
    alignedFovs = []

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

        alignedImage, alignedVessel, alignedFov = alignDriveImage(
            image,
            vesselMask,
            fovMask,
            discCenter
        )

        alignedImages.append(alignedImage.astype(np.float32))
        alignedVessels.append((alignedVessel > 0).astype(np.float32))
        alignedFovs.append((alignedFov > 0).astype(np.float32))

        print(f"Aligned image {imageId}")

    imageSum = np.sum(
        np.stack(alignedImages) *
        np.stack(alignedFovs)[..., None],
        axis=0
    )

    validCount = np.sum(
        np.stack(alignedFovs),
        axis=0
    )

    averageImage = imageSum / np.maximum(
        validCount[..., None], 1
    )

    vesselSum = np.sum(
        np.stack(alignedVessels), axis=0
    )

    probabilityMap = vesselSum / np.maximum(validCount, 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].imshow(averageImage.astype(np.uint8))
    axes[0].set_title("Average Aligned Retina")

    axes[1].imshow(probabilityMap, cmap="hot", vmin=0, vmax=1)
    axes[1].set_title("Preliminary Vessel Probability")

    axes[2].imshow(validCount, cmap="viridis")
    axes[2].set_title("Valid Pixel Count")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

