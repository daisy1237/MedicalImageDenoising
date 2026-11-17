
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

projRoot = Path(__file__).resolve().parent.parent
resultsDir = projRoot / "results"
figuresDir = resultsDir / "final_figures"

methods = [
    "Weak Gaussian",
    "Strong Gaussian",
    "Adaptive V2",
    "Adaptive V4",
    "Noise-Aware Control"
]

metrics = {
    "global_psnr": ("Global PSNR", "PSNR (dB)"),
    "opl_psnr": ("OPL PSNR", "PSNR (dB)"),
    "opl_gradient_error": (
        "OPL Gradient Error",
        "Gradient Error"
    )
}


def generateFinalFigures():
    figuresDir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(
        resultsDir / "adaptive_v4_final_summary.csv"
    )

    noiseLevels = sorted(df["sigma"].unique())

    for metric, (title, ylabel) in metrics.items():
        plt.figure(figsize=(9, 5))

        for method in methods:
            selected = df[df["method"] == method]
            selected = selected.sort_values("sigma")

            plt.plot(
                selected["sigma"],
                selected[metric],
                marker="o",
                linewidth=2,
                label=method
            )

        plt.title(title)
        plt.xlabel("Noise sigma")
        plt.ylabel(ylabel)

        plt.xticks(noiseLevels)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        outputPath = figuresDir / f"{metric}.png"

        plt.savefig(
            outputPath,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()

        print("Saved:", outputPath.name)

    print("\nAll final figures generated.")
