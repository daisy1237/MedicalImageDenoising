
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

projRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projRoot))

from src.spatialNormalization import normalizeMasks
from src.probabilityMap import buildProbabilityMap

"""
    create a synthetic 3D spherical structure using physical coordinates in millimeters
"""
def create_synthetic_volume(size, spacing):
    image = sitk.Image(size, sitk.sitkUInt8)
    image.SetSpacing(spacing)

    z, y, x = np.indices(
        (size[2], size[1], size[0])
    )

    x_mm = x * spacing[0]
    y_mm = y * spacing[1]
    z_mm = z * spacing[2]

    sphere = (
        (x_mm - 32) ** 2
        + (y_mm - 32) ** 2
        + (z_mm - 32) ** 2
    ) <= 15 ** 2

    result = sitk.GetImageFromArray(
        sphere.astype(np.uint8)
    )
    result.CopyInformation(image)

    return result

"""
    test 3D resampling and probability map generation
"""
def runNormaliztionTest():
    volume1 = create_synthetic_volume(
        size=[64, 64, 64],
        spacing=[1.0, 1.0, 1.0]
    )

    volume2 = create_synthetic_volume(
        size=[32, 32, 32],
        spacing=[2.0, 2.0, 2.0]
    )

    normalized_masks = normalizeMasks(
        [volume1, volume2],
        reference=volume1
    )

    probability_map = buildProbabilityMap(
        normalized_masks
    )

    print("===== 3D NORMALIZATION TEST =====")
    print(f"reference size: {volume1.GetSize()}")
    print(f"reference spacing: {volume1.GetSpacing()}")
    print(f"probability map shape: {probability_map.shape}")
    print(f"minimum probability: {probability_map.min():.4f}")
    print(f"maximum probability: {probability_map.max():.4f}")
    print(f"mean probability: {probability_map.mean():.4f}")

    assert all(
        mask.shape == probability_map.shape
        for mask in normalized_masks
    )

    print("3D normalization test passed!")
