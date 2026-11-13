
import csv
import sys
from pathlib import Path
import cv2
from PIL import Image
import numpy as np
projRoot = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(projRoot))
from src.filters import (gaussianFilter,bilateralFilter,nonLocalMeansFilter)
from src.noise import addNoise
from src.metrics import evaluateImage

trainingDir = projRoot / "data" / "DRIVE" / "training"
imagesDir = trainingDir / "images"
fovMasksDir = trainingDir / "mask"
resultsDir = projRoot / "results"
noiseLevels = [10, 20, 30]
randomSeed = 42
methods = ["noisy","gaussian","bilateral", "non local means"]

def loadImage(path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"could not load image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def applyFilters(noisyImage):
    return {
        "noisy": noisyImage,
        "gaussian": gaussianFilter(noisyImage),
        "bilateral": bilateralFilter(noisyImage),
        "non local means": nonLocalMeansFilter(noisyImage)
    }

def runExperiments():
    resultsDir.mkdir(parents=True, exist_ok=True)

    imagePaths = sorted(imagesDir.glob("*.tif"))

    if not imagePaths:
        raise FileNotFoundError(
            f"no DRIVE training images found in: {imagesDir}"
        )

    results = []

    for imagePath in imagePaths:
        original = loadImage(imagePath)

        imageId = imagePath.stem.split("_")[0]
        maskPath = fovMasksDir / f"{imageId}_training_mask.gif"

        if not maskPath.exists():
            raise FileNotFoundError(f"Missing FOV mask: {maskPath}")

        fovMask = np.array(Image.open(maskPath).convert("L"))

        for sigma in noiseLevels:
            seed = randomSeed + int(imagePath.stem.split("_")[0]) * 100 + sigma

            noisyImage = addNoise(original, sigma=sigma, seed=seed)

            filteredImages = applyFilters(noisyImage)

            for method, processed in filteredImages.items():
                metrics = evaluateImage(original, processed, fovMask)
                results.append({
                    "image": imagePath.name,
                    "sigma": sigma,
                    "method": method,
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"]
                })

        print(f"processed: {imagePath.name}")

    saveResults(results)
    saveAverages(results)

    print("\nexperiments completed successfully!")


def saveResults(results):
    outputPath = resultsDir / "filter_results.csv"

    with open(outputPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "sigma", "method", "psnr", "ssim"]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nindividual results saved to: {outputPath}")


def saveAverages(results):
    averages = []

    for sigma in noiseLevels:
        for method in methods:
            selected = [
                result for result in results
                if result["sigma"] == sigma
                and result["method"] == method
            ]

            averages.append({
                "sigma": sigma,
                "method": method,
                "avg_psnr": np.mean([r["psnr"] for r in selected]),
                "avg_ssim": np.mean([r["ssim"] for r in selected])
            })

    outputPath = resultsDir / "filter_averages.csv"

    with open(outputPath, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["sigma", "method", "avg_psnr", "avg_ssim"]
        )

        writer.writeheader()
        writer.writerows(averages)

    print(f"average results saved to: {outputPath}")

    print("\n===== AVERAGE FILTER PERFORMANCE =====")
    print(f"{'sigma':<8} {'method':<20} {'psnr (dB)':<15} {'ssim':<10}")
    print("-" * 55)

    for result in averages:
        print(
            f"{result['sigma']:<8} "
            f"{result['method']:<20} "
            f"{result['avg_psnr']:<15.4f} "
            f"{result['avg_ssim']:<10.4f}"
        )


