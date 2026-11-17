
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

projRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projRoot))

from src.viewOct import buildOPLTrainingMasks
from src.retinalNormalization import (
    normalizeRetinaTwoBoundaries,
    normalizeMaskTwoBoundaries
)

dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"


"""
    build OPL probability map using two-boundary
    anatomical normalization
"""
def buildOPLProbabilityMapV2():
    resultsDir.mkdir(parents=True, exist_ok=True)

    structureCount = None
    validCount = None
    processedSlices = 0

    for subjectId in range(1, 7):
        filePath = dataDir / f"Subject_{subjectId:02d}.mat"
        data = loadmat(filePath)

        images = data["images"]
        manualLayers = data["manualLayers1"]

        height, width, numSlices = images.shape

        if structureCount is None:
            structureCount = np.zeros(
                (height, width), dtype=np.int32
            )
            validCount = np.zeros(
                (height, width), dtype=np.int32
            )

        subjectSlices = 0

        for sliceId in range(numSlices):
            layers = manualLayers[:, :, sliceId]

            structureMask, validMask = buildOPLTrainingMasks(
                layers, (height, width)
            )

            if not np.any(structureMask):
                continue

            try:
                _, coordinates, support = (
                    normalizeRetinaTwoBoundaries(
                        images[:, :, sliceId],
                        layers
                    )
                )
            except ValueError:
                continue

            normalizedStructure = (
                normalizeMaskTwoBoundaries(
                    structureMask, coordinates, support
                )
            )

            normalizedValidity = (
                normalizeMaskTwoBoundaries(
                    validMask, coordinates, support
                ).astype(bool) & support
            )

            # Count only anatomically supported observations
            normalizedStructure = (
                normalizedStructure.astype(bool)
                & normalizedValidity
            )

            structureCount += normalizedStructure.astype(np.int32)
            validCount += normalizedValidity.astype(np.int32)

            processedSlices += 1
            subjectSlices += 1

        print(
            f"Subject {subjectId:02d}: "
            f"{subjectSlices} processed slices"
        )

    if processedSlices == 0:
        raise ValueError("No valid OCT slices processed")

    probabilityMap = np.full(
        structureCount.shape,
        np.nan,
        dtype=np.float32
    )

    np.divide(
        structureCount,
        validCount,
        out=probabilityMap,
        where=validCount > 0
    )

    np.save(
        resultsDir / "opl_probability_map_v4_train.npy",
        probabilityMap
    )

    np.save(
        resultsDir / "opl_valid_count_v4_train.npy",
        validCount
    )

    np.save(
        resultsDir / "opl_structure_count_v4_train.npy",
        structureCount
    )

    print("\n===== OPL PROBABILITY MAP V2 =====")
    print(f"Processed slices: {processedSlices}")
    print(f"Map shape: {probabilityMap.shape}")
    print(f"Supported pixels: {np.sum(validCount > 0)}")
    print(f"Maximum observations: {np.max(validCount)}")

    supported = probabilityMap[np.isfinite(probabilityMap)]

    print(
        f"Probability range: "
        f"{supported.min():.4f} - "
        f"{supported.max():.4f}"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im1 = axes[0].imshow(
        np.ma.masked_invalid(probabilityMap),
        cmap="hot",
        vmin=0,
        vmax=1
    )

    axes[0].set_title("OPL probability map - V2")
    axes[0].axis("off")
    fig.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(
        validCount,
        cmap="viridis",
        vmin=0,
        vmax=88
    )

    axes[1].set_title("Valid observation count - V2")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()

    outputPath = resultsDir / "opl_probability_map_v4_train.png"
    plt.savefig(outputPath, dpi=200)
    plt.show()
    plt.close(fig)

    print(f"\nResults saved to: {resultsDir}")
