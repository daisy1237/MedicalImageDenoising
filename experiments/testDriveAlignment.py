
import cv2
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
from src.driveAlignment import detectOpticDisc,getOpticDiscCenter
from src.viewImage import loadImage, loadMask, imgDir, masks


def testDetection(imageId):
    imagePath = imgDir / f"{imageId}_training.tif"
    fovPath = masks / f"{imageId}_training_mask.gif"

    image = loadImage(imagePath)
    fovMask = loadMask(fovPath)

    center, responseMap = detectOpticDisc(image, fovMask)

    print(f"Image: {imageId}")
    print(f"Detected optic disc center: {center}")

    markedImage = image.copy()

    cv2.circle(
        markedImage,
        center,
        25,
        (0, 255, 0),
        2
    )

    cv2.drawMarker(
        markedImage,
        center,
        (255, 0, 0),
        cv2.MARKER_CROSS,
        20,
        2
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")

    axes[1].imshow(
        np.ma.masked_where(responseMap < 0, responseMap),
        cmap="hot"
    )
    axes[1].set_title("Brightness Response")

    axes[2].imshow(markedImage)
    axes[2].set_title("Detected Optic Disc")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def testAllDetections():
    imageFiles = sorted(imgDir.glob("*_training.tif"))

    fig, axes = plt.subplots(4, 5, figsize=(18, 15))

    for ax, imagePath in zip(axes.flat, imageFiles):
        imageId = imagePath.stem.split("_")[0]

        image = loadImage(imagePath)
        fovMask = loadMask(
            masks / f"{imageId}_training_mask.gif"
        )

        #center, _ = detectOpticDisc(image, fovMask)
        center = getOpticDiscCenter(image, fovMask, imageId)

        markedImage = image.copy()

        cv2.circle(
            markedImage,
            center,
            20,
            (0, 255, 0),
            2
        )

        ax.imshow(markedImage)
        ax.set_title(f"Image {imageId}: {center}")
        ax.axis("off")

        print(f"Image {imageId}: {center}")

    plt.tight_layout()
    plt.show()


def selectOpticDiscCenters():
    projectRoot = Path(__file__).resolve().parent.parent
    outputPath = projectRoot / "results" / "drive_optic_disc_centers.json"
    outputPath.parent.mkdir(parents=True, exist_ok=True)

    if outputPath.exists():
        with open(outputPath, "r") as file:
            centers = json.load(file)
    else:
        centers = {}

    datasets = [
        ("training", projectRoot / "data" / "DRIVE" / "training"),
        ("test", projectRoot / "data" / "DRIVE" / "test")
    ]

    for datasetName, datasetDir in datasets:
        imageFiles = sorted(
            (datasetDir / "images").glob("*.tif")
        )

        for imagePath in imageFiles:
            imageId = imagePath.stem.split("_")[0]
            centerKey = f"{datasetName}_{imageId}"

            if centerKey in centers:
                print(f"Skipping {centerKey}: already saved")
                continue

            if datasetName == "training" and imageId in centers:
                centers[centerKey] = centers[imageId]
                continue

            image = loadImage(imagePath)

            fovPath = (
                datasetDir / "mask" /
                f"{imageId}_{datasetName}_mask.gif"
            )
            fovMask = loadMask(fovPath)

            detectedCenter, _ = detectOpticDisc(image, fovMask)

            fig, ax = plt.subplots(figsize=(8, 7))
            ax.imshow(image)

            ax.scatter(
                detectedCenter[0],
                detectedCenter[1],
                s=130,
                facecolors="none",
                edgecolors="lime",
                linewidths=2
            )

            ax.set_title(
                f"{datasetName} - Image {imageId}\n"
                "Left click: select center | Enter: keep automatic"
            )
            ax.axis("off")

            selectedCenter = [None]

            def onClick(event):
                if event.inaxes != ax or event.button != 1:
                    return

                x = int(round(event.xdata))
                y = int(round(event.ydata))

                height, width = image.shape[:2]

                if not (0 <= x < width and 0 <= y < height):
                    return

                if fovMask[y, x] == 0:
                    return

                selectedCenter[0] = (x, y)
                plt.close(fig)

            def onKey(event):
                if event.key == "enter":
                    selectedCenter[0] = detectedCenter
                    plt.close(fig)

            fig.canvas.mpl_connect(
                "button_press_event", onClick
            )
            fig.canvas.mpl_connect(
                "key_press_event", onKey
            )

            plt.show(block=True)

            if selectedCenter[0] is None:
                print(f"Skipped {centerKey}")
                continue

            centers[centerKey] = list(selectedCenter[0])

            with open(outputPath, "w") as file:
                json.dump(centers, file, indent=4)

            print(f"{centerKey}: {selectedCenter[0]}")

    print(f"\nSaved centers to {outputPath}")
