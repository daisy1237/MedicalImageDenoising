
from pathlib import Path

import numpy as np
from PIL import Image
from datasets import load_dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "DRIVE" / "test" / "rite_manual"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def downloadRiteMasks():
    dataset = load_dataset(
        "Kaphathy/OphGen",
        "DRIVE",
        split="test",
        streaming=True
    )

    savedIds = set()

    for sample in dataset:
        sourcePaths = sample["source_paths"]

        # Identify the corresponding DRIVE test image.
        imagePath = next(
            (
                path for path in sourcePaths
                if path.startswith("test/images/")
                and path.endswith("_test.tif")
            ),
            None
        )

        if imagePath is None:
            continue

        imageId = Path(imagePath).stem.split("_")[0]

        if imageId in savedIds:
            continue

        vesselMask = next(
            (
                item["mask"]
                for item in sample["masks"]
                if item["name"] == "vessel_segmentation (RITE)"
            ),
            None
        )

        if vesselMask is None:
            print(f"Missing RITE mask for {imageId}")
            continue

        maskArray = np.array(
            vesselMask.convert("L")
        )

        # Convert to binary mask.
        binaryMask = (
            (maskArray > 0).astype(np.uint8) * 255
        )

        outputPath = OUTPUT_DIR / f"{imageId}_rite.png"

        Image.fromarray(binaryMask).save(outputPath)

        savedIds.add(imageId)

        print(f"Saved {outputPath.name}")

        if len(savedIds) == 20:
            break

    print(f"\nTotal masks saved: {len(savedIds)}")

    expectedIds = {
        f"{i:02d}" for i in range(1, 21)
    }

    missingIds = sorted(expectedIds - savedIds)

    if missingIds:
        print("Missing image IDs:", missingIds)
    else:
        print("All 20 DRIVE test masks downloaded successfully.")


if __name__ == "__main__":
    downloadRiteMasks()
