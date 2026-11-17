
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from src.viewOct import buildOPLTrainingMasks
from src.retinalNormalization import (
    normalizeRetinaTwoBoundaries,
    normalizeMaskTwoBoundaries
)
from src.adaptiveFilter import adaptiveGaussianV4
from experiments.evaluateAdaptiveBatch import addNoise


projRoot = Path(__file__).resolve().parent.parent
dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"
visualDir = resultsDir / "visual_comparisons"

noiseParameters = {
    0.04: 0,
    0.08: 15,
    0.12: 20
}


def visualizeAdaptiveV4(subjectId=9, sliceId=30):
    visualDir.mkdir(parents=True, exist_ok=True)

    data = loadmat(
        dataDir / f"Subject_{subjectId:02d}.mat"
    )

    image = data["images"][:, :, sliceId]
    layers = data["manualLayers1"][:, :, sliceId]

    clean, coordinates, support = (
        normalizeRetinaTwoBoundaries(image, layers)
    )

    clean = clean.astype(np.float32) / 255.0

    structureMask, validMask = buildOPLTrainingMasks(
        layers, image.shape
    )

    oplMask = normalizeMaskTwoBoundaries(
        structureMask, coordinates, support
    ).astype(bool)

    annotationValidity = normalizeMaskTwoBoundaries(
        validMask, coordinates, support
    ).astype(bool)

    oplMask &= annotationValidity & support

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

        weak, strong, adaptive = (
            adaptiveGaussianV4(
                noisy, probability, support,
                noiseSigma=sigma,
                k=k
            )
        )

        strongError = np.abs(clean - strong)
        adaptiveError = np.abs(clean - adaptive)

        # Use the same intensity range for all images.
        images = [
            clean,
            noisy,
            strong,
            adaptive,
            strongError,
            adaptiveError
        ]

        titles = [
            "Reference OCT",
            "Noisy",
            "Strong Gaussian",
            "Adaptive V4",
            "Strong Gaussian Error",
            "Adaptive V4 Error"
        ]

        fig, axes = plt.subplots(
            2, 3, figsize=(16, 9)
        )

        for index, ax in enumerate(axes.flat):
            isError = index >= 4

            ax.imshow(
                images[index],
                cmap="gray" if not isError else "magma",
                vmin=0,
                vmax=0.25 if isError else 1
            )

            ax.set_title(titles[index])
            ax.axis("off")

        fig.suptitle(
            f"Subject {subjectId:02d} | "
            f"Slice {sliceId} | "
            f"Noise sigma {sigma} | k={k}"
        )

        fig.tight_layout()

        outputPath = visualDir / (
            f"subject_{subjectId:02d}_"
            f"slice_{sliceId}_sigma_{sigma:.2f}.png"
        )

        fig.savefig(
            outputPath,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Saved:", outputPath.name)

        # OPL zoom: use the bounding rectangle
        # of the annotated OPL region.
        ys, xs = np.where(oplMask)

        if len(ys) == 0:
            continue

        yMin = max(0, int(ys.min()) - 12)
        yMax = min(clean.shape[0], int(ys.max()) + 13)

        xMin = max(0, int(xs.min()))
        xMax = min(clean.shape[1], int(xs.max()) + 1)

        fig, axes = plt.subplots(
            1, 3, figsize=(15, 5)
        )

        zoomImages = [clean, strong, adaptive]
        zoomTitles = [
            "Reference OPL",
            "Strong Gaussian OPL",
            "Adaptive V4 OPL"
        ]

        for ax, zoomImage, title in zip(
            axes, zoomImages, zoomTitles
        ):
            ax.imshow(
                zoomImage[yMin:yMax, xMin:xMax],
                cmap="gray",
                vmin=0,
                vmax=1
            )

            ax.set_title(title)
            ax.axis("off")

        fig.suptitle(
            f"OPL comparison | Subject {subjectId:02d} | "
            f"Sigma {sigma}"
        )

        fig.tight_layout()

        zoomPath = visualDir / (
            f"opl_subject_{subjectId:02d}_"
            f"slice_{sliceId}_sigma_{sigma:.2f}.png"
        )

        fig.savefig(
            zoomPath,
            dpi=180,
            bbox_inches="tight"
        )

        plt.close(fig)

        print("Saved:", zoomPath.name)
