
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures" / "drive"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = RESULTS_DIR / "drive_adaptive_test_results.csv"

OUR_METHOD = "Adaptive Smoothed Tuned"

BASELINES = [
    "Strong Gaussian",
    "Noise-Aware Control"
]

METRICS = {
    "vessel_psnr": "Vessel PSNR (dB)",
    "vessel_gradient_error": "Vessel Gradient Error"
}

RANDOM_SEED = 42
BOOTSTRAP_ITERATIONS = 10000


def bootstrapConfidenceInterval(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(RANDOM_SEED)

    bootstrapMeans = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(
            values,
            size=len(values),
            replace=True
        )

        bootstrapMeans.append(np.mean(sample))

    return np.percentile(
        bootstrapMeans, [2.5, 97.5]
    )


def analyzePairedDifferences(df):
    records = []

    for metric, metricLabel in METRICS.items():
        pivot = df.pivot(
            index="image",
            columns="method",
            values=metric
        )

        for baseline in BASELINES:
            paired = pivot[
                [OUR_METHOD, baseline]
            ].dropna()

            differences = (
                paired[OUR_METHOD] -
                paired[baseline]
            ).to_numpy()

            # For gradient error, smaller is better.
            if metric == "vessel_gradient_error":
                improvements = -differences
            else:
                improvements = differences

            ciLow, ciHigh = bootstrapConfidenceInterval(
                improvements
            )

            if np.allclose(improvements, 0):
                pValue = 1.0
            else:
                pValue = wilcoxon(
                    improvements,
                    alternative="two-sided"
                ).pvalue

            records.append({
                "metric": metric,
                "baseline": baseline,
                "n_images": len(improvements),
                "mean_improvement": np.mean(improvements),
                "median_improvement": np.median(improvements),
                "std_improvement": np.std(
                    improvements, ddof=1
                ),
                "ci95_low": ciLow,
                "ci95_high": ciHigh,
                "positive_images": np.sum(
                    improvements > 0
                ),
                "negative_images": np.sum(
                    improvements < 0
                ),
                "wilcoxon_p": pValue
            })

    return pd.DataFrame(records)


def plotMethodComparison(df):
    order = [
        "Weak Gaussian",
        "Strong Gaussian",
        "Noise-Aware Control",
        "Adaptive Direct",
        "Adaptive Smoothed",
        "Adaptive Smoothed Tuned"
    ]

    summary = df.groupby("method")[
        "vessel_psnr"
    ].agg(["mean", "std"])

    summary = summary.loc[order]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        summary.index,
        summary["mean"],
        yerr=summary["std"],
        capsize=4
    )

    ax.set_ylabel("Vessel PSNR (dB)")
    ax.set_title("DRIVE Test - Vessel PSNR Comparison")
    ax.tick_params(axis="x", rotation=25)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "drive_vessel_psnr_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plotIndividualDifferences(df):
    pivot = df.pivot(
        index="image",
        columns="method",
        values="vessel_psnr"
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 5))

    for baseline in BASELINES:
        differences = (
            pivot[OUR_METHOD] -
            pivot[baseline]
        )

        ax.plot(
            differences.index,
            differences.values,
            marker="o",
            label=f"vs {baseline}"
        )

    ax.axhline(0, linestyle="--", color="black")

    ax.set_xlabel("DRIVE Test Image")
    ax.set_ylabel("Vessel PSNR Difference (dB)")
    ax.set_title(
        "Adaptive Smoothed Tuned - Individual Improvements"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "drive_individual_improvements.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plotConfidenceIntervals(statisticsDf):
    psnrStats = statisticsDf[
        statisticsDf["metric"] == "vessel_psnr"
    ]

    fig, ax = plt.subplots(figsize=(8, 4))

    for index, (_, row) in enumerate(
        psnrStats.iterrows()
    ):
        mean = row["mean_improvement"]

        ax.errorbar(
            mean,
            index,
            xerr=[
                [mean - row["ci95_low"]],
                [row["ci95_high"] - mean]
            ],
            fmt="o",
            capsize=5
        )

    ax.set_yticks(
        range(len(psnrStats)),
        psnrStats["baseline"]
    )

    ax.axvline(0, linestyle="--", color="black")

    ax.set_xlabel("Mean Vessel PSNR Improvement (dB)")
    ax.set_title("95% Bootstrap Confidence Intervals")

    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "drive_confidence_intervals.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    df = pd.read_csv(
        INPUT_FILE,
        dtype={"image": str}
    )

    statisticsDf = analyzePairedDifferences(df)

    statisticsDf.to_csv(
        RESULTS_DIR / "drive_statistical_analysis.csv",
        index=False
    )

    plotMethodComparison(df)
    plotIndividualDifferences(df)
    plotConfidenceIntervals(statisticsDf)

    print("\n===== STATISTICAL ANALYSIS =====")
    print(statisticsDf.round(5).to_string(index=False))

    print("\nFigures saved in:")
    print(FIGURES_DIR)

