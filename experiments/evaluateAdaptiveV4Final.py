
import csv
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.viewOct import buildOPLTrainingMasks
from src.retinalNormalization import (
    normalizeRetinaTwoBoundaries,
    normalizeMaskTwoBoundaries
)
from src.adaptiveFilter import (
    adaptiveGaussianV2,
    adaptiveGaussianV4
)
from experiments.evaluateAdaptiveBatch import (
    addNoise,
    prepareMasks,
    calculateResults
)


projRoot = Path(__file__).resolve().parent.parent
dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"

testSubjects = [9, 10]

noiseParameters = {
    0.04: 0,
    0.08: 15,
    0.12: 20
}


def evaluateAdaptiveV4Final():
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

    for subjectId in testSubjects:
        data = loadmat(
            dataDir / f"Subject_{subjectId:02d}.mat"
        )

        images = data["images"]
        manualLayers = data["manualLayers1"]

        processed = 0

        for sliceId in range(images.shape[2]):
            image = images[:, :, sliceId]
            layers = manualLayers[:, :, sliceId]

            structureMask, _ = buildOPLTrainingMasks(
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
                raise ValueError(
                    "Probability map shape mismatch"
                )

            oplMask, gradientMask = prepareMasks(
                layers,
                image.shape,
                coordinates,
                support
            )

            if not np.any(oplMask):
                continue

            if not np.any(gradientMask):
                continue

            for noiseIndex, (sigma, k) in enumerate(
                noiseParameters.items()
            ):
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

                _, _, adaptiveV4 = (
                    adaptiveGaussianV4(
                        noisy, probability, support,
                        noiseSigma=sigma,
                        k=k
                    )
                )

                # Control without spatial probability map.
                # Uses the same average protection weight
                # as V4 for this image.
                noiseFactor = np.exp(-k * sigma)

                meanProbability = float(
                    np.mean(probability[support])
                )

                constantWeight = (
                    meanProbability * noiseFactor
                )

                noiseAwareControl = (
                    constantWeight * weak
                    + (1 - constantWeight) * strong
                )

                noiseAwareControl = noiseAwareControl.astype(np.float32)
                noiseAwareControl[~support] = 0

                methods = {
                    "Weak Gaussian": weak,
                    "Strong Gaussian": strong,
                    "Adaptive V2": adaptiveV2,
                    "Adaptive V4": adaptiveV4,
                    "Noise-Aware Control": noiseAwareControl
                }

                for methodName, filtered in methods.items():
                    metrics = calculateResults(
                        clean,
                        filtered,
                        support,
                        oplMask,
                        gradientMask
                    )

                    results.append({
                        "subject": subjectId,
                        "slice": sliceId,
                        "sigma": sigma,
                        "k": k,
                        "method": methodName,
                        **metrics
                    })

            processed += 1

        print(
            f"Subject {subjectId:02d}: "
            f"{processed} evaluated slices"
        )

    if not results:
        raise ValueError("No valid results")

    outputPath = (
        resultsDir / "adaptive_v4_final_results.csv"
    )

    with open(outputPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== FINAL ADAPTIVE V4 EVALUATION =====")
    print(f"Total evaluations: {len(results)}")

    summary = []

    for sigma, k in noiseParameters.items():
        print(f"\n===== SIGMA: {sigma} | k: {k} =====")

        for method in [
            "Weak Gaussian",
            "Strong Gaussian",
            "Adaptive V2",
            "Adaptive V4",
            "Noise-Aware Control"
        ]:
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
                "k": k,
                "method": method,
                "slices": len(selected),
                **averages
            })

            print(
                f"{method:20s} | "
                f"Global PSNR: {averages['global_psnr']:.4f} | "
                f"OPL PSNR: {averages['opl_psnr']:.4f} | "
                f"OPL MAE: {averages['opl_mae']:.4f} | "
                f"Gradient: {averages['opl_gradient_error']:.4f}"
            )

    summaryPath = (
        resultsDir / "adaptive_v4_final_summary.csv"
    )

    with open(summaryPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys())
        )
        writer.writeheader()
        writer.writerows(summary)

    print("\nResults saved to:")
    print(outputPath)
    print(summaryPath)
