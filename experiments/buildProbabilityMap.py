
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

projRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projRoot))

from src.probabilityMap import buildProbabilityMap


trainingDir = (
    projRoot / "data" / "DRIVE"
    / "training"
)

vesselMaskDir = trainingDir / "1st_manual"
fovMaskDir= trainingDir / "mask"

resultsDir = projRoot / "results"

"""
    load a binary segmentation mask from a gif file
    args:
        path: path to the mask file
    returns:
        boolean numpy array
"""
def loadBinaryMask(path):
    with Image.open(path) as image:
        return np.array(image.convert("L")) > 0

"""
    load vessel segmentation masks and corresponding fov masks
    returns:
        tuple containing vessel masks and valid region masks
"""
def loadDriveMasks():
    vesselMasks = []
    validMasks = []

    maskPaths = sorted(vesselMaskDir.glob("*_manual1.gif"))

    if not maskPaths:
        raise FileNotFoundError("No DRIVE vessel masks found.")

    for mask_path in maskPaths:
        image_id = mask_path.stem.split("_")[0]

        fov_path = (
            fovMaskDir / f"{image_id}_training_mask.gif"
        )

        if not fov_path.exists():
            raise FileNotFoundError(
                f"Missing FOV mask: {fov_path}"
            )

        vesselMasks.append(loadBinaryMask(mask_path))
        validMasks.append(loadBinaryMask(fov_path))

    return vesselMasks, validMasks

"""
    build, save and visualize the statistical vessel probability map
"""
def runProbMap():
    resultsDir.mkdir(parents=True, exist_ok=True)

    vessel_masks, valid_masks = loadDriveMasks()

    probability_map = buildProbabilityMap(vessel_masks, valid_masks)

    np.save(
        resultsDir / "vessel_probability_map.npy",
        probability_map
    )

    print("\n===== VESSEL PROBABILITY MAP =====")
    print(f"number of masks: {len(vessel_masks)}")
    print(f"map shape: {probability_map.shape}")
    print(f"minimum: {probability_map.min():.4f}")
    print(f"maximum: {probability_map.max():.4f}")
    print(f"mean: {probability_map.mean():.4f}")

    plt.figure(figsize=(8, 7))

    plt.imshow(
        probability_map,
        cmap="hot",
        vmin=0,
        vmax=1
    )

    plt.colorbar(label="Vessel probability")
    plt.title("Statistical Vessel Probability Map")
    plt.axis("off")
    plt.tight_layout()

    plt.savefig(
        resultsDir / "vessel_probability_map.png",
        dpi=300
    )

    plt.show()


