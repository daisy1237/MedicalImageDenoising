
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

projRoot = Path(__file__).resolve().parent
trainingDir = projRoot / "data" / "DRIVE" / "training"

imgDir = trainingDir / "images"
manualMasks = trainingDir / "1st_manual"
masks = trainingDir / "mask"

def load_image(path):
    image = cv2.imread(str(path))

    if image is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def load_mask(path):
    return np.array(Image.open(path).convert("L"))

def visualize_sample(image_id):
    image_path = imgDir / f"{image_id}_training.tif"
    vessel_path = manualMasks / f"{image_id}_manual1.gif"
    fov_path = masks / f"{image_id}_training_mask.gif"

    image = load_image(image_path)
    vessel_mask = load_mask(vessel_path)
    fov_mask = load_mask(fov_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original retinal image")

    axes[1].imshow(vessel_mask, cmap="gray")
    axes[1].set_title("Vessel segmentation")

    axes[2].imshow(fov_mask, cmap="gray")
    axes[2].set_title("Field of view mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_sample("21")
