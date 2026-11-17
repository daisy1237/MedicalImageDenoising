
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

projRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projRoot))

from src.viewOct import buildOPLTrainingMasks
from src.retinalNormalization import (
    normalizeRetinaTwoBoundaries,
    normalizeMaskTwoBoundaries
)
from experiments.evaluateOplProbabilityMap import calculateMetrics

dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
testSubjects = [9, 10]


"""
    evaluate V2 probability map on held-out subjects
    using two-boundary normalization
"""
def evaluateOPLProbabilityMapV2():
    probabilityMap = np.load(
        resultsDir / "opl_probability_map_v2.npy"
    )

    validCount = np.load(
        resultsDir / "opl_valid_count_v2.npy"
    )

    supported = (
        np.isfinite(probabilityMap)
        & (validCount >= 5)
    )

    results = []

    for subjectId in testSubjects:
        filePath = dataDir / f"Subject_{subjectId:02d}.mat"
        data = loadmat(filePath)

        images = data["images"]
        manualLayers = data["manualLayers1"]

        height, width, numSlices = images.shape
        processed = 0

        if probabilityMap.shape != (height, width):
            raise ValueError("Probability map shape mismatch")

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
                ).astype(bool)
            )

            normalizedValidity = (
                normalizeMaskTwoBoundaries(
                    validMask, coordinates, support
                ).astype(bool) & support
            )

            normalizedStructure &= normalizedValidity

            evaluationMask = normalizedValidity & supported

            if not np.any(evaluationMask):
                continue

            validPixels = int(np.sum(normalizedValidity))
            evaluatedPixels = int(np.sum(evaluationMask))

            coverage = (
                evaluatedPixels / validPixels
                if validPixels else 0.0
            )

            for threshold in thresholds:
                prediction = (
                    (probabilityMap >= threshold)
                    & supported
                )

                metrics = calculateMetrics(
                    prediction,
                    normalizedStructure,
                    evaluationMask
                )

                if metrics is None:
                    continue

                results.append({
                    "subject": subjectId,
                    "slice": sliceId,
                    "threshold": threshold,
                    "coverage": round(coverage, 4),
                    "dice": round(metrics["dice"], 4),
                    "iou": round(metrics["iou"], 4),
                    "precision": round(metrics["precision"], 4),
                    "recall": round(metrics["recall"], 4),
                    "tp": metrics["tp"],
                    "fp": metrics["fp"],
                    "fn": metrics["fn"]
                })

            processed += 1

        print(
            f"Subject {subjectId:02d}: "
            f"{processed} evaluated slices"
        )

    outputPath = (
        resultsDir / "opl_probability_evaluation_v2.csv"
    )

    fieldnames = [
        "subject", "slice", "threshold", "coverage",
        "dice", "iou", "precision", "recall",
        "tp", "fp", "fn"
    ]

    with open(outputPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== OPL PROBABILITY EVALUATION V2 =====")

    for threshold in thresholds:
        selected = [
            r for r in results
            if r["threshold"] == threshold
        ]

        if not selected:
            continue

        print(f"\nThreshold: {threshold}")

        for metric in [
            "dice", "iou", "precision",
            "recall", "coverage"
        ]:
            average = np.mean(
                [r[metric] for r in selected]
            )
            print(f"Average {metric}: {average:.4f}")

    print(f"\nResults saved to: {outputPath}")
