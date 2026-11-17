
from pathlib import Path
import pandas as pd

projRoot = Path(__file__).resolve().parent.parent
resultsDir = projRoot / "results"


def analyzeFinalVariability():
    df = pd.read_csv(
        resultsDir / "adaptive_v4_final_results.csv"
    )

    paired = df.pivot(
        index=["subject", "slice", "sigma"],
        columns="method",
        values="opl_psnr"
    )

    paired["difference"] = (
        paired["Adaptive V4"]
        - paired["Noise-Aware Control"]
    )

    summary = paired.groupby(level="sigma")[
        "difference"
    ].agg(["mean", "std", "count"])

    summary["wins"] = paired.groupby(level="sigma")[
        "difference"
    ].apply(lambda values: (values > 0).sum())

    print("\n===== V4 VS NOISE-AWARE CONTROL =====")
    print(summary.round(4))

    outputPath = (
        resultsDir / "adaptive_v4_variability.csv"
    )

    summary.to_csv(outputPath)

    print("\nSaved:", outputPath)
