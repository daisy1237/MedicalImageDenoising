
import csv
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

from src.viewOct import buildOPLTrainingMasks
from src.retinalNormalization import (
    normalizeRetinaTwoBoundaries,
    normalizeMaskTwoBoundaries
)
from src.adaptiveFilter import (
    adaptiveGaussianV2,
    adaptiveBilateralGaussian
)
from experiments.evaluateAdaptiveFilterV2 import maskedPSNR
from src.metrics import maskedMAE, gradientError


projRoot = Path(__file__).resolve().parent.parent
dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"

testSubjects = [9, 10]
noiseLevels = [0.04, 0.08, 0.12]


def addNoise(clean, support, sigma, seed):
    rng = np.random.default_rng(seed)

    noise = rng.normal(
        0, sigma, clean.shape
    ).astype(np.float32)

    noisy = clean.copy()

    noisy[support] = np.clip(
        clean[support] + noise[support],
        0, 1
    )

    return noisy


def prepareMasks(layers, imageShape, coordinates, support):
    structureMask, validMask = buildOPLTrainingMasks(
        layers, imageShape
    )

    oplMask = normalizeMaskTwoBoundaries(
        structureMask, coordinates, support
    ).astype(bool)

    validMask = normalizeMaskTwoBoundaries(
        validMask, coordinates, support
    ).astype(bool)

    oplMask &= validMask & support

    # Sobel uses a 3x3 neighborhood.
    # Exclude pixels whose neighborhood is invalid.
    interior = cv2.erode(
        support.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
        borderType=cv2.BORDER_CONSTANT
    ).astype(bool)

    gradientMask = oplMask & interior

    return oplMask, gradientMask


def calculateResults(reference, result, support,
                     oplMask, gradientMask):
    return {
        "global_psnr": maskedPSNR(
            reference, result, support
        ),
        "opl_psnr": maskedPSNR(
            reference, result, oplMask
        ),
        "opl_mae": maskedMAE(
            reference, result, oplMask
        ),
        "opl_gradient_error": gradientError(
            reference, result, gradientMask
        )
    }


def evaluateAdaptiveBatch():
    resultsDir.mkdir(parents=True, exist_ok=True)

    probability = np.load(
        resultsDir / "opl_probability_map_v2.npy"
    )

    validCount = np.load(
        resultsDir / "opl_valid_count_v2.npy"
    )

    probability = np.where(
        (validCount >= 5) & np.isfinite(probability),
        probability,
        0.0
    ).astype(np.float32)

    probability = np.clip(probability, 0, 1)

    results = []

    for subjectId in testSubjects:
        data = loadmat(
            dataDir / f"Subject_{subjectId:02d}.mat"
        )

        images = data["images"]
        manualLayers = data["manualLayers1"]

        height, width, numSlices = images.shape
        processed = 0

        for sliceId in range(numSlices):
            layers = manualLayers[:, :, sliceId]

            structureMask, _ = buildOPLTrainingMasks(
                layers, (height, width)
            )

            if not np.any(structureMask):
                continue

            try:
                clean, coordinates, support = (
                    normalizeRetinaTwoBoundaries(
                        images[:, :, sliceId],
                        layers
                    )
                )
            except ValueError:
                continue

            if not np.any(support):
                continue

            if clean.shape != probability.shape:
                raise ValueError(
                    "Probability map shape mismatch"
                )

            clean = clean.astype(np.float32) / 255.0

            oplMask, gradientMask = prepareMasks(
                layers,
                (height, width),
                coordinates,
                support
            )

            if not np.any(oplMask):
                continue

            if not np.any(gradientMask):
                continue

            for noiseIndex, sigma in enumerate(noiseLevels):
                seed = (
                    subjectId * 100000
                    + sliceId * 100
                    + noiseIndex
                )

                noisy = addNoise(
                    clean, support, sigma, seed
                )

                weak, strong, adaptiveV2 = (
                    adaptiveGaussianV2(
                        noisy, probability, support
                    )
                )

                bilateral, _, adaptiveV3 = (
                    adaptiveBilateralGaussian(
                        noisy, probability, support
                    )
                )

                # Control: same mean blend weight,
                # without spatial anatomical guidance.
                meanWeight = float(
                    np.mean(probability[support])
                )

                constantBlend = (
                    meanWeight * bilateral
                    + (1 - meanWeight) * strong
                )

                constantBlend[~support] = 0

                methods = {
                    "Noisy": noisy,
                    "Weak Gaussian": weak,
                    "Strong Gaussian": strong,
                    "Bilateral": bilateral,
                    "Adaptive V2": adaptiveV2,
                    "Adaptive V3": adaptiveV3,
                    "Constant Blend": constantBlend
                }

                for methodName, filtered in methods.items():
                    metrics = calculateResults(
                        clean, filtered, support,
                        oplMask, gradientMask
                    )

                    results.append({
                        "subject": subjectId,
                        "slice": sliceId,
                        "sigma": sigma,
                        "method": methodName,
                        **metrics
                    })

            processed += 1

        print(
            f"Subject {subjectId:02d}: "
            f"{processed} evaluated slices"
        )

    if not results:
        raise ValueError("No valid evaluation results")

    outputPath = (
        resultsDir / "adaptive_batch_results.csv"
    )

    fieldnames = list(results[0].keys())

    with open(outputPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== ADAPTIVE BATCH EVALUATION =====")
    print(f"Total evaluations: {len(results)}")

    methods = [
        "Noisy",
        "Weak Gaussian",
        "Strong Gaussian",
        "Bilateral",
        "Adaptive V2",
        "Adaptive V3",
        "Constant Blend"
    ]

    summary = []

    for sigma in noiseLevels:
        print(f"\n===== NOISE SIGMA: {sigma} =====")

        for method in methods:
            selected = [
                row for row in results
                if row["sigma"] == sigma
                and row["method"] == method
            ]

            if not selected:
                continue

            averages = {}

            for metric in [
                "global_psnr",
                "opl_psnr",
                "opl_mae",
                "opl_gradient_error"
            ]:
                averages[metric] = float(
                    np.mean([
                        row[metric] for row in selected
                    ])
                )

            summary.append({
                "sigma": sigma,
                "method": method,
                "slices": len(selected),
                **averages
            })

            print(
                f"{method:18s} | "
                f"Global PSNR: {averages['global_psnr']:.4f} | "
                f"OPL PSNR: {averages['opl_psnr']:.4f} | "
                f"Gradient: {averages['opl_gradient_error']:.4f}"
            )

    summaryPath = (
        resultsDir / "adaptive_batch_summary.csv"
    )

    with open(summaryPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\nResults saved to:")
    print(outputPath)
    print(summaryPath)
