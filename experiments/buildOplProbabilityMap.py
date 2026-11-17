
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

projRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projRoot))

from src.viewOct import buildOPLTrainingMasks
from src.retinalNormalization import flattenOCT, flattenMask

dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"


"""
    build a statistical probability map for the OPL
    using anatomically normalized OCT images
"""
def buildOPLProbabilityMap():
    resultsDir.mkdir(parents=True, exist_ok=True)

    structureCount = None
    validCount = None

    processedSlices = 0

    for subjectId in range(1, 9):
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
                _, shifts, validReference = flattenOCT(
                    images[:, :, sliceId],
                    layers
                )
            except ValueError:
                continue

            flattenedStructure = flattenMask(
                structureMask, shifts, validReference
            )

            flattenedValidity = flattenMask(
                validMask, shifts, validReference
            )

            # Accumulate only supported anatomical observations.
            structureCount += (
                flattenedStructure.astype(np.int32)
                * flattenedValidity.astype(np.int32)
            )
            validCount += flattenedValidity.astype(np.int32)

            processedSlices += 1
            subjectSlices += 1

        print(
            f"Subject {subjectId:02d}: "
            f"{subjectSlices} processed slices"
        )

    probabilityMap = np.full(
        structureCount.shape, np.nan, dtype=np.float32
    )

    np.divide(
        structureCount,
        validCount,
        out=probabilityMap,
        where=validCount > 0
    )

    np.save(
        resultsDir / "opl_probability_map.npy",
        probabilityMap
    )

    np.save(
        resultsDir / "opl_valid_count.npy",
        validCount
    )

    np.save(
        resultsDir / "opl_structure_count.npy",
        structureCount
    )

    print("\n===== OPL PROBABILITY MAP =====")
    print(f"Processed slices: {processedSlices}")
    print(f"Map shape: {probabilityMap.shape}")
    print(f"Supported pixels: {np.sum(validCount > 0)}")

    supported = probabilityMap[np.isfinite(probabilityMap)]

    if supported.size:
        print(
            f"Probability range: "
            f"{supported.min():.4f} - "
            f"{supported.max():.4f}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    maskedProbability = np.ma.masked_invalid(
        probabilityMap
    )

    im = axes[0].imshow(
        maskedProbability,
        cmap="hot",
        vmin=0,
        vmax=1
    )

    axes[0].set_title("OPL probability map")
    axes[0].axis("off")
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(
        validCount,
        cmap="viridis"
    )

    axes[1].set_title("Valid observation count")
    axes[1].axis("off")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()

    outputPath = resultsDir / "opl_probability_map.png"
    plt.savefig(outputPath, dpi=200)
    plt.show()
    plt.close(fig)

    print(f"Results saved to: {resultsDir}")
