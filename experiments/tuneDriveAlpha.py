
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.evaluateDriveProbabilityMap import (
    loadAlignedSamples,
    buildMaps
)

from experiments.evaluateDriveAdaptive import (
    addNoise,
    gaussianFilter,
    calculatePSNR,
    calculateGradientError,
    WEAK_KERNEL,
    WEAK_SIGMA,
    STRONG_KERNEL,
    STRONG_SIGMA,
    NOISE_SIGMA
)

from src.viewImage import (
    loadImage,
    loadMask,
    imgDir,
    masks
)

from src.driveAlignment import (
    getOpticDiscCenter,
    alignDriveImage
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

ALPHA_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 5]
RANDOM_SEED = 42


def adaptiveFilterWithAlpha(weak, strong, probabilityMap, alpha):
    weight = np.clip(alpha * probabilityMap, 0, 1)

    return (
        weight[..., None] * weak +
        (1 - weight[..., None]) * strong
    )


def tuneAlpha():
    samples = loadAlignedSamples()
    rng = np.random.default_rng(RANDOM_SEED)

    results = []

    for testIndex, testSample in enumerate(samples):
        imageId = testSample["id"]

        trainingSamples = [
            sample
            for index, sample in enumerate(samples)
            if index != testIndex
        ]

        _, smoothedMap, validCount = buildMaps(
            trainingSamples
        )

        image = loadImage(
            imgDir / f"{imageId}_training.tif"
        )

        fovMask = loadMask(
            masks / f"{imageId}_training_mask.gif"
        )

        center = getOpticDiscCenter(
            image, fovMask, imageId
        )

        alignedImage, _, alignedFov = alignDriveImage(
            image,
            np.zeros(image.shape[:2], dtype=np.uint8),
            fovMask,
            center
        )

        clean = alignedImage.astype(np.float32) / 255.0

        validMask = (
            (alignedFov > 0) &
            (validCount > 0)
        )

        vesselMask = (
            (testSample["vessels"] > 0) &
            validMask
        )

        backgroundMask = validMask & ~vesselMask

        noisy = addNoise(clean, NOISE_SIGMA, rng)

        weak = gaussianFilter(
            noisy, WEAK_KERNEL, WEAK_SIGMA
        )

        strong = gaussianFilter(
            noisy, STRONG_KERNEL, STRONG_SIGMA
        )

        for alpha in ALPHA_VALUES:
            filtered = adaptiveFilterWithAlpha(
                weak, strong, smoothedMap, alpha
            )

            results.append({
                "image": imageId,
                "alpha": alpha,
                "global_psnr": calculatePSNR(
                    clean, filtered, validMask
                ),
                "vessel_psnr": calculatePSNR(
                    clean, filtered, vesselMask
                ),
                "background_psnr": calculatePSNR(
                    clean, filtered, backgroundMask
                ),
                "vessel_gradient_error": calculateGradientError(
                    clean, filtered, vesselMask
                )
            })

        print(f"Evaluated training image {imageId}")

    resultsDf = pd.DataFrame(results)

    summaryDf = resultsDf.groupby("alpha")[
        [
            "global_psnr",
            "vessel_psnr",
            "background_psnr",
            "vessel_gradient_error"
        ]
    ].agg(["mean", "std"])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    resultsDf.to_csv(
        RESULTS_DIR / "drive_alpha_tuning_results.csv",
        index=False
    )

    summaryDf.to_csv(
        RESULTS_DIR / "drive_alpha_tuning_summary.csv"
    )

    print("\n===== ALPHA TUNING RESULTS =====")
    print(summaryDf.round(4).to_string())

    # Choose alpha by mean vessel PSNR on training only.
    vesselMeans = resultsDf.groupby("alpha")[
        "vessel_psnr"
    ].mean()

    bestAlpha = float(vesselMeans.idxmax())

    print(f"\nBest alpha (vessel PSNR): {bestAlpha}")

    return bestAlpha

