
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

projRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projRoot))

from src.viewOct import buildOPLTrainingMasks
from src.retinalNormalization import flattenOCT, flattenMask

dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"

thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]
testSubjects = [9, 10]


"""
    calculate segmentation metrics using only
    valid annotated positions
"""
def calculateMetrics(prediction, groundTruth, validMask):
    valid = validMask.astype(bool)

    pred = prediction[valid].astype(bool)
    truth = groundTruth[valid].astype(bool)

    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))

    if tp + fp + fn == 0:
        return None

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    dice = 2 * tp / (2 * tp + fp + fn)
    iou = tp / (tp + fp + fn)

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


"""
    evaluate the OPL probability map on unseen subjects
"""
def evaluateOPLProbabilityMap():
    probabilityPath = resultsDir / "opl_probability_map.npy"
    validCountPath = resultsDir / "opl_valid_count.npy"

    probabilityMap = np.load(probabilityPath)
    validCount = np.load(validCountPath)

    # Do not make predictions where the training map
    # has no statistical support.
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
                _, shifts, validReference = flattenOCT(
                    images[:, :, sliceId],
                    layers
                )
            except ValueError:
                continue

            flattenedStructure = flattenMask(
                structureMask, shifts, validReference
            ).astype(bool)

            flattenedValidity = flattenMask(
                validMask, shifts, validReference
            ).astype(bool)

            # Evaluate only positions where both
            # ground truth and training support exist.
            evaluationMask = flattenedValidity & supported

            if not np.any(evaluationMask):
                continue

            validPixels = int(np.sum(flattenedValidity))
            evaluatedPixels = int(np.sum(evaluationMask))

            coverage = (
                evaluatedPixels / validPixels
                if validPixels else 0.0
            )

            for threshold in thresholds:
                prediction = (
                    probabilityMap >= threshold
                ) & supported

                metrics = calculateMetrics(
                    prediction,
                    flattenedStructure,
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

    outputPath = resultsDir / "opl_probability_evaluation.csv"

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

    print("\n===== OPL PROBABILITY EVALUATION =====")

    for threshold in thresholds:
        selected = [
            r for r in results
            if r["threshold"] == threshold
        ]

        if not selected:
            continue

        print(f"\nThreshold: {threshold}")
        print(f"Average Dice: "
              f"{np.mean([r['dice'] for r in selected]):.4f}")
        print(f"Average IoU: "
              f"{np.mean([r['iou'] for r in selected]):.4f}")
        print(f"Average Precision: "
              f"{np.mean([r['precision'] for r in selected]):.4f}")
        print(f"Average Recall: "
              f"{np.mean([r['recall'] for r in selected]):.4f}")
        print(f"Average coverage: "
              f"{np.mean([r['coverage'] for r in selected]):.4f}")

    print(f"\nResults saved to: {outputPath}")
