
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.vesselThickness import classifyVessels
from src.viewImage import loadImage, loadMask
from src.driveAlignment import (
    getOpticDiscCenter,
    alignDriveImage
)


BEST_ALPHA = 0.5

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = PROJECT_ROOT / "data" / "DRIVE" / "test"
RESULTS_DIR = PROJECT_ROOT / "results"

NOISE_SIGMA = 0.08
RANDOM_SEED = 42

WEAK_KERNEL = (3, 3)
WEAK_SIGMA = 0.6

STRONG_KERNEL = (7, 7)
STRONG_SIGMA = 1.8


def addNoise(image, sigma, rng):
    noise = rng.normal(
        0, sigma, image.shape
    ).astype(np.float32)

    return np.clip(image + noise, 0, 1)


def gaussianFilter(image, kernel, sigma):
    return cv2.GaussianBlur(
        image, kernel, sigmaX=sigma
    )


def adaptiveFilter(weak, strong, probabilityMap):
    weight = probabilityMap[..., None]

    return (
        weight * weak +
        (1 - weight) * strong
    )


def calculatePSNR(reference, filtered, mask):
    valid = mask > 0

    if not np.any(valid):
        return np.nan

    mse = np.mean(
        (reference[valid] - filtered[valid]) ** 2
    )

    if mse == 0:
        return float("inf")

    return 10 * np.log10(1.0 / mse)


def calculateGradientError(reference, filtered, mask):
    refGray = reference[:, :, 1]
    filteredGray = filtered[:, :, 1]

    refX = cv2.Sobel(
        refGray, cv2.CV_32F, 1, 0
    )
    refY = cv2.Sobel(
        refGray, cv2.CV_32F, 0, 1
    )

    filteredX = cv2.Sobel(
        filteredGray, cv2.CV_32F, 1, 0
    )
    filteredY = cv2.Sobel(
        filteredGray, cv2.CV_32F, 0, 1
    )

    error = np.sqrt(
        (refX - filteredX) ** 2 +
        (refY - filteredY) ** 2
    )

    valid = mask > 0

    if not np.any(valid):
        return np.nan

    return float(np.mean(error[valid]))


def loadTestVesselMask(imageId):
    maskPath = (
        TEST_DIR / "rite_manual" /
        f"{imageId}_rite.png"
    )

    if not maskPath.exists():
        raise FileNotFoundError(
            f"Missing RITE vessel mask: {maskPath}"
        )

    return loadMask(maskPath)


def evaluateDriveAdaptive():
    directMap = np.load(
        RESULTS_DIR / "drive_probability_direct.npy"
    )

    smoothedMap = np.load(
        RESULTS_DIR / "drive_probability_smoothed.npy"
    )

    validCount = np.load(
        RESULTS_DIR / "drive_valid_count.npy"
    )

    rng = np.random.default_rng(RANDOM_SEED)
    results = []

    imageFiles = sorted(
        (TEST_DIR / "images").glob("*_test.tif")
    )

    if not imageFiles:
        raise FileNotFoundError(
            "No DRIVE test images found."
        )

    for imagePath in imageFiles:
        imageId = imagePath.stem.split("_")[0]

        image = loadImage(imagePath)

        fovMask = loadMask(
            TEST_DIR / "mask" /
            f"{imageId}_test_mask.gif"
        )

        vesselMask = loadTestVesselMask(imageId)

        center = getOpticDiscCenter(
            image,
            fovMask,
            imageId,
            dataset="test"
        )

        alignedImage, alignedVessels, alignedFov = (
            alignDriveImage(
                image,
                vesselMask,
                fovMask,
                center
            )
        )

        clean = (
            alignedImage.astype(np.float32) / 255.0
        )

        if clean.shape[:2] != directMap.shape:
            raise ValueError(
                f"Map/image shape mismatch for {imageId}: "
                f"{directMap.shape} vs {clean.shape[:2]}"
            )

        validMask = (
            (alignedFov > 0) &
            (validCount > 0)
        )

        vesselRegion = (
            (alignedVessels > 0) &
            validMask
        )

        backgroundRegion = (
            validMask &
            ~vesselRegion
        )

        # Classify vessels by estimated local thickness.
        # These masks contain vessel centerline pixels.
        thinVessels, thickVessels = classifyVessels(
            alignedVessels
        )

        thinVessels = (
            thinVessels &
            validMask
        )

        thickVessels = (
            thickVessels &
            validMask
        )

        noisy = addNoise(
            clean, NOISE_SIGMA, rng
        )

        weak = gaussianFilter(
            noisy,
            WEAK_KERNEL,
            WEAK_SIGMA
        )

        strong = gaussianFilter(
            noisy,
            STRONG_KERNEL,
            STRONG_SIGMA
        )

        adaptiveDirect = adaptiveFilter(
            weak,
            strong,
            directMap
        )

        adaptiveSmoothed = adaptiveFilter(
            weak,
            strong,
            smoothedMap
        )

        adaptiveSmoothedTuned = adaptiveFilter(
            weak,
            strong,
            np.clip(
                BEST_ALPHA * smoothedMap,
                0,
                1
            )
        )

        controlWeight = float(
            np.mean(smoothedMap[validMask])
        )

        noiseAwareControl = (
            controlWeight * weak +
            (1 - controlWeight) * strong
        )

        methods = {
            "Weak Gaussian": weak,
            "Strong Gaussian": strong,
            "Adaptive Direct": adaptiveDirect,
            "Adaptive Smoothed": adaptiveSmoothed,
            "Noise-Aware Control": noiseAwareControl,
            "Adaptive Smoothed Tuned": adaptiveSmoothedTuned
        }

        for methodName, filtered in methods.items():
            row = {
                "image": imageId,
                "method": methodName,

                "global_psnr": calculatePSNR(
                    clean,
                    filtered,
                    validMask
                ),

                "vessel_psnr": calculatePSNR(
                    clean,
                    filtered,
                    vesselRegion
                ),

                "background_psnr": calculatePSNR(
                    clean,
                    filtered,
                    backgroundRegion
                ),

                "vessel_gradient_error": (
                    calculateGradientError(
                        clean,
                        filtered,
                        vesselRegion
                    )
                ),

                "thin_vessel_psnr": calculatePSNR(
                    clean,
                    filtered,
                    thinVessels
                ),

                "thick_vessel_psnr": calculatePSNR(
                    clean,
                    filtered,
                    thickVessels
                )
            }

            results.append(row)

        print(f"Evaluated TEST image {imageId}")

    resultsDf = pd.DataFrame(results)

    metrics = [
        "global_psnr",
        "vessel_psnr",
        "background_psnr",
        "vessel_gradient_error",
        "thin_vessel_psnr",
        "thick_vessel_psnr"
    ]

    summaryDf = (
        resultsDf.groupby("method")[metrics]
        .agg(["mean", "std"])
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    resultsDf.to_csv(
        RESULTS_DIR / "drive_adaptive_test_results.csv",
        index=False
    )

    summaryDf.to_csv(
        RESULTS_DIR / "drive_adaptive_test_summary.csv"
    )

    print("\n===== DRIVE TEST RESULTS =====")
    print(
        summaryDf.round(4).to_string()
    )


if __name__ == "__main__":
    evaluateDriveAdaptive()
