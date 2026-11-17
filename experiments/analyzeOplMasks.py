
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
    count continuous groups of valid columns
"""
def countSegments(validColumns):
    padded = np.pad(
        validColumns.astype(np.int8),
        (1, 1),
        constant_values=0
    )

    changes = np.diff(padded)

    return int(np.sum(changes == 1))


"""
    analyze the thickness and continuity of
    the selected retinal region
"""
def analyzeOPLMasks():
    resultsDir.mkdir(parents=True, exist_ok=True)

    results = []

    for subjectId in range(1, 11):
        filePath = dataDir / f"Subject_{subjectId:02d}.mat"
        data = loadmat(filePath)

        images = data["images"]
        manualLayers = data["manualLayers1"]

        height, width, numSlices = images.shape

        for sliceId in range(numSlices):
            layers = manualLayers[:, :, sliceId]

            mask, validColumns = buildOPLMask(
                layers,
                (height, width)
            )

            if not np.any(validColumns):
                continue

            # thickness of the actual binary mask
            thickness = np.sum(mask, axis=0)
            validThickness = thickness[validColumns]

            # number of separate annotated segments
            segments = countSegments(validColumns)

            # missing columns between first and last
            # valid anatomical annotation
            indices = np.where(validColumns)[0]
            first = int(indices[0])
            last = int(indices[-1])

            internalGaps = int(
                np.sum(~validColumns[first:last + 1])
            )

            results.append({
                "subject": subjectId,
                "slice": sliceId,
                "valid_columns": int(np.sum(validColumns)),
                "valid_percentage": round(
                    100 * np.mean(validColumns), 2
                ),
                "mean_thickness": round(
                    float(np.mean(validThickness)), 2
                ),
                "min_thickness": int(np.min(validThickness)),
                "max_thickness": int(np.max(validThickness)),
                "segments": segments,
                "internal_gap_columns": internalGaps,
                "mask_pixels": int(np.sum(mask))
            })

        print(f"Processed subject {subjectId:02d}")

    outputPath = resultsDir / "opl_mask_analysis.csv"

    fieldnames = [
        "subject",
        "slice",
        "valid_columns",
        "valid_percentage",
        "mean_thickness",
        "min_thickness",
        "max_thickness",
        "segments",
        "internal_gap_columns",
        "mask_pixels"
    ]

    with open(outputPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(results)

    print("\n===== OPL MASK ANALYSIS =====")
    print(f"Analyzed slices: {len(results)}")

    if not results:
        print("No valid masks found.")
        return

    print(
        "Average thickness: "
        f"{np.mean([r['mean_thickness'] for r in results]):.2f} pixels"
    )

    print(
        "Average segments: "
        f"{np.mean([r['segments'] for r in results]):.2f}"
    )

    print(
        "Average internal gaps: "
        f"{np.mean([r['internal_gap_columns'] for r in results]):.2f}"
    )

    print(
        "Slices with multiple segments: "
        f"{sum(r['segments'] > 1 for r in results)}"
    )

    print("\n===== MOST FRAGMENTED MASKS =====")

    mostFragmented = sorted(
        results,
        key=lambda r: (
            -r["segments"],
            -r["internal_gap_columns"]
        )
    )[:5]

    for result in mostFragmented:
        print(
            f"Subject {result['subject']:02d}, "
            f"slice {result['slice']}: "
            f"{result['segments']} segments, "
            f"{result['internal_gap_columns']} gap columns"
        )

    print(f"\nResults saved to: {outputPath}")
