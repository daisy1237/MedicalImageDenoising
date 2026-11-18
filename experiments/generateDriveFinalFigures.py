
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.evaluateDriveAdaptive import (
    addNoise,
    gaussianFilter,
    adaptiveFilter,
    loadTestVesselMask,
    WEAK_KERNEL,
    WEAK_SIGMA,
    STRONG_KERNEL,
    STRONG_SIGMA,
    NOISE_SIGMA,
    RANDOM_SEED,
    BEST_ALPHA
)

from src.viewImage import loadImage, loadMask
from src.driveAlignment import (
    getOpticDiscCenter,
    alignDriveImage
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_DIR = PROJECT_ROOT / "data" / "DRIVE" / "test"

FIGURES_DIR = (
    RESULTS_DIR / "figures" / "drive" / "final"
)

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUR_METHOD = "Adaptive Smoothed Tuned"

BASELINES = [
    "Strong Gaussian",
    "Noise-Aware Control"
]

METRICS = {
    "thin_vessel_psnr": "Thin-vessel centerline PSNR",
    "thick_vessel_psnr": "Thick-vessel centerline PSNR"
}

BOOTSTRAP_ITERATIONS = 10000
BOOTSTRAP_SEED = 42


def bootstrapCI(values):
    values = np.asarray(values, dtype=float)

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    samples = rng.choice(
        values,
        size=(BOOTSTRAP_ITERATIONS, len(values)),
        replace=True
    )

    means = samples.mean(axis=1)

    return np.percentile(means, [2.5, 97.5])


def analyzeVesselThickness(df):
    records = []

    for metric in METRICS:
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

            ciLow, ciHigh = bootstrapCI(differences)

            records.append({
                "metric": metric,
                "baseline": baseline,
                "n_images": len(differences),
                "mean_improvement": differences.mean(),
                "median_improvement": np.median(differences),
                "ci95_low": ciLow,
                "ci95_high": ciHigh,
                "positive_images": np.sum(differences > 0),
                "negative_images": np.sum(differences < 0)
            })

    result = pd.DataFrame(records)

    result.to_csv(
        RESULTS_DIR /
        "drive_vessel_thickness_statistics.csv",
        index=False
    )

    return result


def plotVesselThicknessPSNR(df):
    methods = [
        "Strong Gaussian",
        "Noise-Aware Control",
        "Adaptive Smoothed",
        OUR_METHOD
    ]

    summary = (
        df.groupby("method")[
            list(METRICS.keys())
        ]
        .mean()
        .loc[methods]
    )

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(
        x - width / 2,
        summary["thin_vessel_psnr"],
        width,
        label="Thin vessels"
    )

    ax.bar(
        x + width / 2,
        summary["thick_vessel_psnr"],
        width,
        label="Thick vessels"
    )

    ax.set_xticks(x, methods, rotation=15)
    ax.set_ylabel("Centerline PSNR (dB)")
    ax.set_title(
        "DRIVE Test - Vessel Thickness Comparison"
    )
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "vessel_thickness_psnr.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plotThinVesselImprovements(df):
    pivot = df.pivot(
        index="image",
        columns="method",
        values="thin_vessel_psnr"
    ).sort_index()

    fig, ax = plt.subplots(figsize=(11, 5))

    for baseline in BASELINES:
        difference = (
            pivot[OUR_METHOD] -
            pivot[baseline]
        )

        ax.plot(
            difference.index,
            difference.values,
            marker="o",
            label=f"vs {baseline}"
        )

    ax.axhline(
        0, color="black", linestyle="--"
    )

    ax.set_xlabel("DRIVE Test Image")
    ax.set_ylabel("Thin-vessel PSNR improvement (dB)")
    ax.set_title(
        "Adaptive Smoothed Tuned - Thin Vessel Improvements"
    )

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "thin_vessel_improvements.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def plotConfidenceIntervals(statisticsDf):
    fig, ax = plt.subplots(figsize=(10, 5))

    labels = []

    for index, row in statisticsDf.iterrows():
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

        vesselType = (
            "Thin" if row["metric"] ==
            "thin_vessel_psnr" else "Thick"
        )

        labels.append(
            f"{vesselType} vs {row['baseline']}"
        )

    ax.set_yticks(
        range(len(labels)), labels
    )

    ax.axvline(
        0, color="black", linestyle="--"
    )

    ax.set_xlabel("Mean PSNR improvement (dB)")
    ax.set_title(
        "95% Bootstrap Confidence Intervals"
    )

    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR /
        "vessel_thickness_confidence_intervals.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def getTestImageResults(targetId="11"):
    smoothedMap = np.load(
        RESULTS_DIR / "drive_probability_smoothed.npy"
    )

    validCount = np.load(
        RESULTS_DIR / "drive_valid_count.npy"
    )

    rng = np.random.default_rng(RANDOM_SEED)

    for number in range(1, 21):
        imageId = f"{number:02d}"

        image = loadImage(
            TEST_DIR / "images" /
            f"{imageId}_test.tif"
        )

        fov = loadMask(
            TEST_DIR / "mask" /
            f"{imageId}_test_mask.gif"
        )

        vesselMask = loadTestVesselMask(imageId)

        center = getOpticDiscCenter(
            image, fov, imageId, dataset="test"
        )

        alignedImage, _, alignedFov = alignDriveImage(
            image, vesselMask, fov, center
        )

        clean = alignedImage.astype(np.float32) / 255.0
        noisy = addNoise(clean, NOISE_SIGMA, rng)

        if imageId != targetId:
            continue

        weak = gaussianFilter(
            noisy, WEAK_KERNEL, WEAK_SIGMA
        )

        strong = gaussianFilter(
            noisy, STRONG_KERNEL, STRONG_SIGMA
        )

        validMask = (
            (alignedFov > 0) &
            (validCount > 0)
        )

        controlWeight = float(
            np.mean(smoothedMap[validMask])
        )

        control = (
            controlWeight * weak +
            (1 - controlWeight) * strong
        )

        tunedMap = np.clip(
            BEST_ALPHA * smoothedMap,
            0, 1
        )

        adaptive = adaptiveFilter(
            weak, strong, tunedMap
        )

        return clean, {
            "Strong Gaussian": strong,
            "Noise-Aware Control": control,
            OUR_METHOD: adaptive
        }, validMask

    raise ValueError(
        f"Test image {targetId} not found."
    )


def plotAbsoluteErrors(clean, methods, validMask):
    errors = {
        name: np.mean(
            np.abs(clean - filtered),
            axis=2
        )
        for name, filtered in methods.items()
    }

    displayMax = max(
        np.percentile(error[validMask], 99)
        for error in errors.values()
    )

    fig, axes = plt.subplots(
        1, len(errors), figsize=(15, 5)
    )

    for ax, (name, error) in zip(
        axes, errors.items()
    ):
        displayed = np.where(
            validMask, error, np.nan
        )

        im = ax.imshow(
            displayed,
            cmap="magma",
            vmin=0,
            vmax=displayMax
        )

        ax.set_title(name)
        ax.axis("off")

    fig.colorbar(
        im, ax=axes,
        fraction=0.025,
        pad=0.02
    )

    fig.savefig(
        FIGURES_DIR /
        "drive_11_absolute_errors.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    return errors


def plotErrorDifference(errors, validMask):
    difference = (
        errors["Noise-Aware Control"] -
        errors[OUR_METHOD]
    )

    # Positive: our method has lower absolute error.
    # Negative: the control has lower absolute error.
    limit = np.percentile(
        np.abs(difference[validMask]),
        99
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    im = ax.imshow(
        np.where(validMask, difference, np.nan),
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    ax.set_title(
        "Error Reduction: Adaptive vs Control"
    )
    ax.axis("off")

    fig.colorbar(
        im, ax=ax,
        label="Control error - Adaptive error"
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR /
        "drive_11_error_difference.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)


def main():
    df = pd.read_csv(
        RESULTS_DIR / "drive_adaptive_test_results.csv",
        dtype={"image": str}
    )

    statisticsDf = analyzeVesselThickness(df)

    plotVesselThicknessPSNR(df)
    plotThinVesselImprovements(df)
    plotConfidenceIntervals(statisticsDf)

    clean, methods, validMask = (
        getTestImageResults("11")
    )

    errors = plotAbsoluteErrors(
        clean, methods, validMask
    )

    plotErrorDifference(
        errors, validMask
    )

    print("\n===== VESSEL THICKNESS STATISTICS =====")
    print(
        statisticsDf.round(5).to_string(index=False)
    )

    print("\nFinal figures saved in:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()
