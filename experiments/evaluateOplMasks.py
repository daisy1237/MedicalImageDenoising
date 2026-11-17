
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

projRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projRoot))

from src.viewOct import buildOPLMask

dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"


"""
    evaluate the availability and quality of OPL masks
    for all subjects and manually annotated OCT slices
"""
def evaluateOPLMasks():
    resultsDir.mkdir(parents=True, exist_ok=True)

    results = []

    for subjectId in range(1, 11):
        filePath = dataDir / f"Subject_{subjectId:02d}.mat"

        if not filePath.exists():
            raise FileNotFoundError(
                f"Missing OCT file: {filePath}"
            )

        data = loadmat(filePath)

        images = data["images"]
        manualLayers = data["manualLayers1"]

        height, width, numSlices = images.shape

        validSlices = 0

        for sliceId in range(numSlices):
            layers = manualLayers[:, :, sliceId]

            mask, validColumns = buildOPLMask(
                layers,
                (height, width)
            )

            numValidColumns = int(np.sum(validColumns))
            numPixels = int(np.sum(mask))

            # skip slices without valid annotations
            if numValidColumns == 0:
                continue

            validSlices += 1

            results.append({
                "subject": subjectId,
                "slice": sliceId,
                "valid_columns": numValidColumns,
                "valid_percentage": round(
                    100 * numValidColumns / width, 2
                ),
                "mask_pixels": numPixels
            })

        print(
            f"Subject {subjectId:02d}: "
            f"{validSlices} annotated slices"
        )

    outputPath = resultsDir / "opl_mask_evaluation.csv"

    with open(outputPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "subject",
                "slice",
                "valid_columns",
                "valid_percentage",
                "mask_pixels"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print("\n===== OPL MASK EVALUATION =====")
    print(f"Total valid slices: {len(results)}")

    if results:
        print(
            "Average valid columns: "
            f"{np.mean([r['valid_columns'] for r in results]):.2f}"
        )

        print(
            "Average mask pixels: "
            f"{np.mean([r['mask_pixels'] for r in results]):.2f}"
        )

    print(f"\nResults saved to: {outputPath}")
