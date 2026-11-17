
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from src.viewOct import buildOPLTrainingMasks
from src.retinalNormalization import (
    normalizeRetinaTwoBoundaries,
    normalizeMaskTwoBoundaries
)
from src.adaptiveFilter import adaptiveGaussianV2


projRoot = Path(__file__).resolve().parent.parent
dataDir = projRoot / "data" / "2015_BOE_Chiu"
resultsDir = projRoot / "results"


"""
    calculate PSNR only on valid pixels
"""
def maskedPSNR(reference, filtered, mask):
    error = (
        reference[mask] - filtered[mask]
    ) ** 2

    mse = np.mean(error)

    if mse == 0:
        return float("inf")

    return 10 * np.log10(1.0 / mse)


"""
    evaluate one OCT slice
"""
def evaluateAdaptiveFilterV2(subjectId=9, sliceId=30):
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
    )

    rng = np.random.default_rng(42)

    noisy = clean.copy()

    noise = rng.normal(
        0, 0.08, clean.shape
    ).astype(np.float32)

    noisy[support] = np.clip(
        clean[support] + noise[support],
        0, 1
    )

    weak, strong, adaptive = adaptiveGaussianV2(
        noisy, probability, support
    )

    print("\n===== ADAPTIVE FILTER EVALUATION =====")
    print(f"Subject: {subjectId:02d}")
    print(f"Slice: {sliceId}")
    print(f"Valid pixels: {np.sum(support)}")
    print(f"OPL pixels: {np.sum(oplMask)}")

    methods = {
        "Noisy": noisy,
        "Weak Gaussian": weak,
        "Strong Gaussian": strong,
        "Adaptive V2": adaptive
    }

    print("\nGLOBAL PSNR:")
    for name, result in methods.items():
        print(
            f"{name}: "
            f"{maskedPSNR(clean, result, support):.4f} dB"
        )

    print("\nOPL PSNR:")
    for name, result in methods.items():
        print(
            f"{name}: "
            f"{maskedPSNR(clean, result, oplMask):.4f} dB"
        )
