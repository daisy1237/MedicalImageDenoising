
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk

projRoot = Path(__file__).resolve().parent.parent
dataDir = projRoot / "data" / "HumanBonyLabyrinth"


"""
    load a CT volume and its anatomical labels
    args:
        specimenId: specimen identifier (e.g. F01)
    returns:
        original SimpleITK volume and segmentation labels
"""
def loadVolume(specimenId):
    specimenDir = dataDir / specimenId / "CT"

    imagePath = specimenDir / f"{specimenId}_CT_RAW.nii"
    labelsPath = specimenDir / f"{specimenId}_CT_LABELS.nii"

    if not imagePath.exists() or not labelsPath.exists():
        raise FileNotFoundError(
            f"Missing CT files for specimen {specimenId}"
        )

    image = sitk.ReadImage(str(imagePath))
    labels = sitk.ReadImage(str(labelsPath))

    return image, labels


"""
    display three orthogonal slices of a CT volume
    and their corresponding segmentation labels
"""
def visualizeVolume(specimenId="F01"):
    image, labels = loadVolume(specimenId)

    volume = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(labels)

    print(f"\n===== SPECIMEN {specimenId} =====")
    print(f"volume size: {image.GetSize()}")
    print(f"voxel spacing: {image.GetSpacing()}")
    print(f"volume shape: {volume.shape}")
    print(f"intensity range: {volume.min()} - {volume.max()}")
    print(f"label values: {np.unique(mask)}")

    z, y, x = np.array(volume.shape) // 2

    imageSlices = [
        volume[z, :, :],
        volume[:, y, :],
        volume[:, :, x]
    ]

    maskSlices = [
        mask[z, :, :],
        mask[:, y, :],
        mask[:, :, x]
    ]

    titles = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i in range(3):
        axes[0, i].imshow(imageSlices[i], cmap="gray")
        axes[0, i].set_title(f"{titles[i]} - CT")

        axes[1, i].imshow(maskSlices[i], cmap="nipy_spectral")
        axes[1, i].set_title(f"{titles[i]} - Labels")

        for row in range(2):
            axes[row, i].axis("off")

    plt.suptitle(f"Human Bony Labyrinth - {specimenId}")
    plt.tight_layout()
    plt.show()


"""
    inspect anatomical structures in a micro-CT volume
    args:
        specimenId: specimen identifier
    returns:
        none
"""
def inspectFineStructures(specimenId="F01"):
    specimenDir = dataDir / specimenId / "uCT"

    imagePath = specimenDir / f"{specimenId}_uCT_RAW.nii"
    labelsPath = specimenDir / f"{specimenId}_uCT_LABELS.nii"

    if not imagePath.exists() or not labelsPath.exists():
        raise FileNotFoundError(
            f"Missing micro-CT files for {specimenId}"
        )

    image = sitk.ReadImage(str(imagePath))
    labels = sitk.ReadImage(str(labelsPath))

    volume = sitk.GetArrayFromImage(image)
    mask = sitk.GetArrayFromImage(labels)

    if volume.shape != mask.shape:
        raise ValueError("Image and label dimensions do not match")

    print(f"\n===== MICRO-CT {specimenId} =====")
    print(f"volume size: {image.GetSize()}")
    print(f"voxel spacing: {image.GetSpacing()}")
    print(f"volume shape: {volume.shape}")

    values, counts = np.unique(mask, return_counts=True)

    print("\n===== LABEL VALUES =====")
    for value, count in zip(values, counts):
        print(f"label {value}: {count} voxels")

    # find the region containing anatomical labels
    positions = np.where(mask > 0)

    if positions[0].size == 0:
        raise ValueError("No anatomical labels found")

    minCoords = np.array([p.min() for p in positions])
    maxCoords = np.array([p.max() for p in positions])

    margin = 15

    start = np.maximum(minCoords - margin, 0)
    end = np.minimum(maxCoords + margin + 1, volume.shape)

    slices = tuple(
        slice(int(start[i]), int(end[i]))
        for i in range(3)
    )

    croppedVolume = volume[slices]
    croppedMask = mask[slices]

    z, y, x = np.array(croppedVolume.shape) // 2

    imageSlices = [
        croppedVolume[z, :, :],
        croppedVolume[:, y, :],
        croppedVolume[:, :, x]
    ]

    maskSlices = [
        croppedMask[z, :, :],
        croppedMask[:, y, :],
        croppedMask[:, :, x]
    ]

    titles = ["Axial", "Coronal", "Sagittal"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    for i in range(3):
        axes[0, i].imshow(imageSlices[i], cmap="gray")
        axes[0, i].set_title(f"{titles[i]} - micro-CT")

        axes[1, i].imshow(imageSlices[i], cmap="gray")

        binaryMask = maskSlices[i] > 0

        if np.any(binaryMask):
            axes[1, i].contour(
                binaryMask.astype(float),
                levels=[0.5],
                colors="red",
                linewidths=0.8
            )

        axes[1, i].set_title(
            f"{titles[i]} - segmentation contours"
        )

        for row in range(2):
            axes[row, i].axis("off")

    plt.suptitle(f"Fine Structure Inspection - {specimenId}")
    plt.tight_layout()
    plt.show()


"""
    inspect all segmentation labels in a micro-CT volume
    args:
        specimenId: specimen identifier
    returns:
        none
"""
def inspectLabels(specimenId="F01"):
    specimenDir = dataDir / specimenId / "uCT"
    labelsPath = specimenDir / f"{specimenId}_uCT_LABELS.nii"

    if not labelsPath.exists():
        raise FileNotFoundError(
            f"Missing micro-CT labels for {specimenId}"
        )

    labels = sitk.ReadImage(str(labelsPath))
    mask = sitk.GetArrayFromImage(labels)

    values, counts = np.unique(mask, return_counts=True)

    print(f"\n===== LABEL ANALYSIS: {specimenId} =====")
    print(f"Volume shape: {mask.shape}")
    print(f"Voxel spacing: {labels.GetSpacing()}")

    for value, count in zip(values, counts):
        print(f"\nLabel {value}: {count} voxels")

        if value == 0:
            continue

        binaryMask = mask == value
        positions = np.where(binaryMask)

        minCoords = [int(p.min()) for p in positions]
        maxCoords = [int(p.max()) for p in positions]

        print(f"  Bounding box min (z,y,x): {minCoords}")
        print(f"  Bounding box max (z,y,x): {maxCoords}")
