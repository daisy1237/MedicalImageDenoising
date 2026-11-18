
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score

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

SMOOTH_SIGMA = 12


def loadAlignedSamples():
    samples = []

    imageFiles = sorted(imgDir.glob("*_training.tif"))

    for imagePath in imageFiles:
        imageId = imagePath.stem.split("_")[0]

        image = loadImage(imagePath)

        vesselMask = loadMask(
            manualMasks / f"{imageId}_manual1.gif"
        )

        fovMask = loadMask(
            masks / f"{imageId}_training_mask.gif"
        )

        center = getOpticDiscCenter(image, fovMask, imageId)

        _, alignedVessels, alignedFov = alignDriveImage(
            image, vesselMask, fovMask, center
        )

        samples.append({
            "id": imageId,
            "vessels": (alignedVessels > 0).astype(np.float32),
            "fov": (alignedFov > 0).astype(np.float32)
        })

        print(f"Loaded image {imageId}")

    return samples


def buildMaps(samples):
    vesselSum = np.sum(
        [sample["vessels"] * sample["fov"] for sample in samples],
        axis=0
    )

    validCount = np.sum(
        [sample["fov"] for sample in samples],
        axis=0
    )

    directMap = vesselSum / np.maximum(validCount, 1)

    smoothVesselSum = cv2.GaussianBlur(
        vesselSum, (0, 0), SMOOTH_SIGMA
    )

    smoothValidCount = cv2.GaussianBlur(
        validCount, (0, 0), SMOOTH_SIGMA
    )

    smoothedMap = smoothVesselSum / np.maximum(
        smoothValidCount, 1e-6
    )

    directMap[validCount == 0] = 0
    smoothedMap[validCount == 0] = 0

    return directMap, smoothedMap, validCount


def evaluateMap(probabilityMap, vesselMask, validMask):
    validPixels = validMask > 0

    trueLabels = vesselMask[validPixels].astype(np.uint8)
    predictions = probabilityMap[validPixels]

    if np.unique(trueLabels).size < 2:
        return np.nan, np.nan

    auc = roc_auc_score(trueLabels, predictions)
    ap = average_precision_score(trueLabels, predictions)

    return auc, ap


def evaluateLeaveOneOut():
    samples = loadAlignedSamples()

    results = []

    for testIndex, testSample in enumerate(samples):
        trainingSamples = [
            sample for index, sample in enumerate(samples)
            if index != testIndex
        ]

        directMap, smoothedMap, validCount = buildMaps(
            trainingSamples
        )

        # Evaluate only where both the held-out image
        # and the training map have valid support.
        validMask = (
            (testSample["fov"] > 0) &
            (validCount > 0)
        )

        for method, probabilityMap in [
            ("Direct", directMap),
            ("Smoothed", smoothedMap)
        ]:
            auc, ap = evaluateMap(
                probabilityMap,
                testSample["vessels"],
                validMask
            )

            results.append({
                "image": testSample["id"],
                "method": method,
                "roc_auc": auc,
                "average_precision": ap
            })

        print(f"Evaluated image {testSample['id']}")

    resultsDf = pd.DataFrame(results)

    summaryDf = resultsDf.groupby("method")[
        ["roc_auc", "average_precision"]
    ].agg(["mean", "std"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    resultsDf.to_csv(
        RESULTS_DIR / "drive_probability_loo_results.csv",
        index=False
    )

    summaryDf.to_csv(
        RESULTS_DIR / "drive_probability_loo_summary.csv"
    )

    print("\n===== LEAVE-ONE-OUT RESULTS =====")
    print(summaryDf.round(4))

    return resultsDf

