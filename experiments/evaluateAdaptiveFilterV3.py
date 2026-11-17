
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
    adaptiveBilateralGaussian
)
from src.metrics import maskedMAE, gradientError
from experiments.evaluateAdaptiveFilterV2 import maskedPSNR


projRoot = Path(__file__).resolve().parent.parent
dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"


def evaluateAdaptiveFilterV3(subjectId=9, sliceId=30):
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
        resultsDir / "opl_probability_map_v2.npy"
    )

    validCount = np.load(
        resultsDir / "opl_valid_count_v2.npy"
    )

    probability = np.where(
        validCount >= 5,
        np.nan_to_num(probability, nan=0.0),
        0.0
    ).astype(np.float32)

    rng = np.random.default_rng(42)

    noisy = clean.copy()

    noise = rng.normal(
        0, 0.08, clean.shape
    ).astype(np.float32)

    noisy[support] = np.clip(
        clean[support] + noise[support],
        0, 1
    )

    weak, strong, adaptiveV2 = adaptiveGaussianV2(
        noisy, probability, support
    )

    bilateral, _, adaptiveV3 = (
        adaptiveBilateralGaussian(
            noisy, probability, support
        )
    )

    methods = {
        "Noisy": noisy,
        "Weak Gaussian": weak,
        "Strong Gaussian": strong,
        "Bilateral": bilateral,
        "Adaptive V2": adaptiveV2,
        "Adaptive V3": adaptiveV3
    }

    print("\n===== ADAPTIVE FILTER V3 EVALUATION =====")
    print(f"Subject: {subjectId:02d}")
    print(f"Slice: {sliceId}")
    print(f"Valid pixels: {np.sum(support)}")
    print(f"OPL pixels: {np.sum(oplMask)}")

    for name, result in methods.items():
        print(f"\n{name}")

        print(
            f"Global PSNR: "
            f"{maskedPSNR(clean, result, support):.4f}"
        )

        print(
            f"OPL PSNR: "
            f"{maskedPSNR(clean, result, oplMask):.4f}"
        )

        print(
            f"OPL MAE: "
            f"{maskedMAE(clean, result, oplMask):.4f}"
        )

        print(
            f"OPL Gradient Error: "
            f"{gradientError(clean, result, oplMask):.4f}"
        )
