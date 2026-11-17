
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np
from src.retinalNormalization import flattenOCT, flattenMask
from src.retinalNormalization import (
    normalizeRetinaTwoBoundaries,
    normalizeMaskTwoBoundaries
)
projRoot = Path(__file__).resolve().parent.parent
dataDir = projRoot / "data" / "2015_BOE_Chiu"


"""
    inspect the variables stored in a Duke DME OCT MATLAB file
    args:
        subjectId: subject identifier (1-10)
"""
def inspectOCT(subjectId=1):
    filePath = dataDir / f"Subject_{subjectId:02d}.mat"

    if not filePath.exists():
        raise FileNotFoundError(
            f"Missing OCT file: {filePath}"
        )

    data = loadmat(filePath)

    print(f"\n===== OCT SUBJECT {subjectId:02d} =====")

    for name, value in data.items():
        if name.startswith("__"):
            continue

        print(f"\nVariable: {name}")
        print(f"  Type: {type(value).__name__}")
        print(f"  Shape: {getattr(value, 'shape', 'N/A')}")
        print(f"  Dtype: {getattr(value, 'dtype', 'N/A')}")


"""
    visualize an OCT B-scan and its manually annotated layers
    args:
        subjectId: subject identifier (1-10)
        sliceId: B-scan index (0-60)
"""
def visualizeOCT(subjectId=1, sliceId=30):
    filePath = dataDir / f"Subject_{subjectId:02d}.mat"

    data = loadmat(filePath)

    image = data["images"][:, :, sliceId]
    layers = data["manualLayers1"][:, :, sliceId]

    print(f"\n===== OCT VISUALIZATION =====")
    print(f"Subject: {subjectId:02d}")
    print(f"Slice: {sliceId}")
    print(f"Image shape: {image.shape}")
    print(f"Layer shape: {layers.shape}")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original OCT image")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray")

    x = np.arange(image.shape[1])

    for layerIndex in range(layers.shape[0]):
        y = layers[layerIndex]

        axes[1].plot(
            x,
            y,
            linewidth=1,
            label=f"Boundary {layerIndex + 1}"
        )

    axes[1].set_title("Manual retinal layer boundaries")
    axes[1].axis("off")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=4,
        fontsize=8
    )

    plt.tight_layout()
    plt.show()


"""
    build a binary mask of the entire OPL region
    using manual boundaries 4 and 5
    args:
        layers: array of shape (8, width)
        imageShape: (height, width)
    returns:
        binary OPL mask and valid column mask
"""
def buildOPLMask(layers, imageShape):
    height, width = imageShape

    upper = layers[3]
    lower = layers[4]

    validColumns = (
        np.isfinite(upper)
        & np.isfinite(lower)
        & (upper < lower)
        & (upper >= 1)
        & (lower <= height)
    )

    mask = np.zeros((height, width), dtype=np.uint8)

    # MATLAB coordinates are 1-based
    for x in np.where(validColumns)[0]:
        y1 = max(0, int(np.floor(upper[x])) - 1)
        y2 = min(height, int(np.ceil(lower[x])))

        mask[y1:y2, x] = 1

    return mask, validColumns


"""
    visualize the complete OPL region on an OCT image
"""
def visualizeOPL(subjectId=1, sliceId=30):
    filePath = dataDir / f"Subject_{subjectId:02d}.mat"
    data = loadmat(filePath)

    image = data["images"][:, :, sliceId]
    layers = data["manualLayers1"][:, :, sliceId]

    mask, validColumns = buildOPLMask(
        layers, image.shape
    )

    print("\n===== OPL MASK =====")
    print(f"Subject: {subjectId:02d}")
    print(f"Slice: {sliceId}")
    print(f"Valid columns: {np.sum(validColumns)} / {image.shape[1]}")
    print(f"OPL pixels: {np.sum(mask)}")

    if not np.any(mask):
        raise ValueError(
            "No valid OPL annotation found in this slice"
        )

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].imshow(image, cmap="gray")
    axes[0].set_title("Original OCT")

    axes[1].imshow(image, cmap="gray")
    overlay = np.ma.masked_where(mask == 0, mask)
    axes[1].imshow(
        overlay,
        cmap="autumn",
        alpha=0.55,
        vmin=0,
        vmax=1
    )
    axes[1].set_title("OPL region overlay")

    axes[2].imshow(mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Binary OPL mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


"""
    analyze why selected retinal boundaries
    cannot form a valid anatomical region
"""
def inspectInvalidOPL(subjectId=5, sliceId=23):
    filePath = dataDir / f"Subject_{subjectId:02d}.mat"
    data = loadmat(filePath)

    image = data["images"][:, :, sliceId]
    layers = data["manualLayers1"][:, :, sliceId]

    upper = layers[3]
    lower = layers[4]

    height, width = image.shape

    finite = np.isfinite(upper) & np.isfinite(lower)

    missing = ~finite

    inverted = (
        finite
        & (upper >= lower)
    )

    outside = (
        finite
        & ~inverted
        & ((upper < 1) | (lower > height))
    )

    valid = finite & ~inverted & ~outside

    print(f"\n===== INVALID OPL ANALYSIS =====")
    print(f"Subject: {subjectId:02d}")
    print(f"Slice: {sliceId}")
    print(f"Total columns: {width}")
    print(f"Missing/non-finite: {np.sum(missing)}")
    print(f"Inverted/equal: {np.sum(inverted)}")
    print(f"Outside image: {np.sum(outside)}")
    print(f"Valid: {np.sum(valid)}")

    assert (
        np.sum(missing)
        + np.sum(inverted)
        + np.sum(outside)
        + np.sum(valid)
        == width
    )


"""
    build structure and validity masks
    for the selected retinal layer
"""
def buildOPLTrainingMasks(layers, imageShape):
    structureMask, validColumns = buildOPLMask(
        layers, imageShape
    )

    height, width = imageShape

    validMask = np.broadcast_to(
        validColumns[np.newaxis, :],
        (height, width)
    ).copy().astype(np.uint8)

    return structureMask, validMask


"""
    visualize anatomical flattening of an OCT image
    and its corresponding OPL mask
"""
def visualizeFlattenedOCT(subjectId=1, sliceId=30):
    filePath = dataDir / f"Subject_{subjectId:02d}.mat"
    data = loadmat(filePath)

    image = data["images"][:, :, sliceId]
    layers = data["manualLayers1"][:, :, sliceId]

    structureMask, validMask = buildOPLTrainingMasks(
        layers, image.shape
    )

    flattenedImage, shifts, validReference = flattenOCT(
        image, layers
    )

    flattenedStructure = flattenMask(
        structureMask, shifts, validReference
    )

    flattenedValidity = flattenMask(
        validMask, shifts, validReference
    )

    print("\n===== RETINAL FLATTENING =====")
    print(f"Subject: {subjectId:02d}")
    print(f"Slice: {sliceId}")
    print(f"Valid reference columns: {np.sum(validReference)}")
    print(f"OPL pixels before: {np.sum(structureMask)}")
    print(f"OPL pixels after: {np.sum(flattenedStructure)}")
    print(f"Valid pixels after: {np.sum(flattenedValidity)}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original OCT")

    axes[0, 1].imshow(flattenedImage, cmap="gray")
    axes[0, 1].set_title("Flattened OCT")

    axes[1, 0].imshow(structureMask, cmap="gray")
    axes[1, 0].set_title("Original OPL mask")

    axes[1, 1].imshow(flattenedStructure, cmap="gray")
    axes[1, 1].set_title("Flattened OPL mask")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)


"""
    compare original OCT with two-boundary
    anatomical normalization
"""
def visualizeTwoBoundaryNormalization(subjectId=1, sliceId=30):
    filePath = dataDir / f"Subject_{subjectId:02d}.mat"
    data = loadmat(filePath)

    image = data["images"][:, :, sliceId]
    layers = data["manualLayers1"][:, :, sliceId]

    structureMask, validMask = buildOPLTrainingMasks(
        layers, image.shape
    )

    normalizedImage, coordinates, support = (
        normalizeRetinaTwoBoundaries(image, layers)
    )

    normalizedStructure = normalizeMaskTwoBoundaries(
        structureMask, coordinates, support
    )

    normalizedValidity = normalizeMaskTwoBoundaries(
        validMask, coordinates, support
    )

    # Only count pixels with valid anatomical support
    normalizedValidity = (
        normalizedValidity.astype(bool) & support
    )

    normalizedStructure[~normalizedValidity] = 0

    print("\n===== TWO-BOUNDARY NORMALIZATION =====")
    print(f"Subject: {subjectId:02d}")
    print(f"Slice: {sliceId}")
    print(f"Original OPL pixels: {np.sum(structureMask)}")
    print(f"Normalized OPL pixels: {np.sum(normalizedStructure)}")
    print(f"Valid normalized pixels: {np.sum(normalizedValidity)}")
    print(f"Supported columns: {np.sum(np.any(support, axis=0))}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Original OCT")

    axes[0, 1].imshow(normalizedImage, cmap="gray")
    axes[0, 1].set_title("Two-boundary normalized OCT")

    axes[1, 0].imshow(structureMask, cmap="gray")
    axes[1, 0].set_title("Original OPL mask")

    axes[1, 1].imshow(
        normalizedStructure, cmap="gray"
    )
    axes[1, 1].set_title("Normalized OPL mask")

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    plt.close(fig)
