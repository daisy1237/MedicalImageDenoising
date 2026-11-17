
from pathlib import Path
import csv

import numpy as np
from scipy.io import loadmat

from src.viewOct import buildOPLTrainingMasks
from src.retinalNormalization import (
    normalizeRetinaTwoBoundaries,
    normalizeMaskTwoBoundaries
)
from src.adaptiveFilter import adaptiveGaussianV4
from experiments.evaluateAdaptiveFilterV2 import maskedPSNR


projRoot = Path(__file__).resolve().parent.parent
dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"

validationSubjects = [7, 8]
noiseLevels = [0.04, 0.08, 0.12]
kValues = [0, 5, 10, 15, 20, 30]


def evaluateAdaptiveV4():
    probability = np.load(
        resultsDir / "opl_probability_map_v4_train.npy"
    )

    validCount = np.load(
        resultsDir / "opl_valid_count_v4_train.npy"
    )

    probability = np.where(
        (validCount >= 5) & np.isfinite(probability),
        probability,
        0.0
    ).astype(np.float32)

    probability = np.clip(probability, 0, 1)

    results = []

    for subjectId in validationSubjects:
        data = loadmat(
            dataDir / f"Subject_{subjectId:02d}.mat"
        )

        images = data["images"]
        manualLayers = data["manualLayers1"]

        processed = 0

        for sliceId in range(images.shape[2]):
            image = images[:, :, sliceId]
            layers = manualLayers[:, :, sliceId]

            structureMask, validMask = buildOPLTrainingMasks(
                layers, image.shape
            )

            if not np.any(structureMask):
                continue

            try:
                clean, coordinates, support = (
                    normalizeRetinaTwoBoundaries(
                        image, layers
                    )
                )
            except ValueError:
                continue

            if not np.any(support):
                continue

            clean = clean.astype(np.float32) / 255.0

            if clean.shape != probability.shape:
                raise ValueError("Probability map shape mismatch")

            oplMask = normalizeMaskTwoBoundaries(
                structureMask, coordinates, support
            ).astype(bool)

            annotationValidity = normalizeMaskTwoBoundaries(
                validMask, coordinates, support
            ).astype(bool)

            oplMask &= annotationValidity & support

            if not np.any(oplMask):
                continue

            for noiseIndex, sigma in enumerate(noiseLevels):
                seed = (
                    subjectId * 100000
                    + sliceId * 100
                    + noiseIndex
                )

                rng = np.random.default_rng(seed)

                noise = rng.normal(
                    0, sigma, clean.shape
                ).astype(np.float32)

                noisy = clean.copy()

                noisy[support] = np.clip(
                    clean[support] + noise[support],
                    0, 1
                )

                for k in kValues:
                    weak, strong, adaptive = (
                        adaptiveGaussianV4(
                            noisy, probability, support,
                            noiseSigma=sigma,
                            k=k
                        )
                    )

                    results.append({
                        "subject": subjectId,
                        "slice": sliceId,
                        "sigma": sigma,
                        "k": k,
                        "global_psnr": maskedPSNR(
                            clean, adaptive, support
                        ),
                        "opl_psnr": maskedPSNR(
                            clean, adaptive, oplMask
                        )
                    })

            processed += 1

        print(
            f"Subject {subjectId:02d}: "
            f"{processed} evaluated slices"
        )

    if not results:
        raise ValueError("No valid evaluation results")

    outputPath = resultsDir / "adaptive_v4_validation.csv"

    with open(outputPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== ADAPTIVE V4 VALIDATION =====")

    summary = []

    for sigma in noiseLevels:
        print(f"\nNOISE SIGMA: {sigma}")

        for k in kValues:
            selected = [
                row for row in results
                if row["sigma"] == sigma and row["k"] == k
            ]

            if not selected:
                continue

            globalPSNR = np.mean([
                row["global_psnr"] for row in selected
            ])

            oplPSNR = np.mean([
                row["opl_psnr"] for row in selected
            ])

            summary.append({
                "sigma": sigma,
                "k": k,
                "global_psnr": float(globalPSNR),
                "opl_psnr": float(oplPSNR)
            })

            print(
                f"k={k:2d} | "
                f"Global PSNR: {globalPSNR:.4f} | "
                f"OPL PSNR: {oplPSNR:.4f}"
            )

    summaryPath = (
        resultsDir / "adaptive_v4_validation_summary.csv"
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
