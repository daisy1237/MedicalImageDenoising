
import cv2
import numpy as np

"""
  estimate the optic disc center using local brightness
  args:
      image: rgb retinal image
      fovMask: binary field-of-view mask
  returns:
      (centerX, centerY), responseMap
"""
def detectOpticDisc(image, fovMask):

    height, width = image.shape[:2]

    redChannel = image[:, :, 0].astype(np.float32)

    validMask = (fovMask > 0).astype(np.float32)

    kernelSize = 51

    brightnessSum = cv2.GaussianBlur(
        redChannel * validMask,
        (kernelSize, kernelSize),
        0
    )

    validWeight = cv2.GaussianBlur(
        validMask,
        (kernelSize, kernelSize),
        0
    )

    responseMap = brightnessSum / np.maximum(validWeight, 1e-6)

    margin = 35

    distanceMap = cv2.distanceTransform(
        validMask.astype(np.uint8),
        cv2.DIST_L2,
        5
    )

    candidateMask = distanceMap > margin

    if not np.any(candidateMask):
        raise ValueError("No valid optic disc candidates found.")

    responseMap[~candidateMask] = -1

    _, _, _, maxLocation = cv2.minMaxLoc(responseMap)

    centerX, centerY = maxLocation

    return (centerX, centerY), responseMap


MANUAL_OPTIC_DISC_CENTERS = {
    "21": (391, 26),
    "22": (432, 28),
    "23": (432, 28),
    "24": (432, 28),
    "25": (432, 28),
    "26": (432, 28),
    "27": (432, 28),
    "28": (432, 28),
    "29": (432, 28),
    "30": (432, 28),
    "31": (432, 28),
    "32": (432, 28),
    "33": (432, 28),
    "34": (432, 28),
    "35": (432, 28),
    "36": (432, 28),
    "37": (432, 28),
    "38": (432, 28),
    "39": (432, 28),
    "40": (432, 28),
}

import json
from pathlib import Path


def getOpticDiscCenter(image, fovMask, imageId, dataset="training"):
    projectRoot = Path(__file__).resolve().parent.parent
    centersPath = projectRoot / "results" / "drive_optic_disc_centers.json"

    if centersPath.exists():
        with open(centersPath, "r") as file:
            centers = json.load(file)

        centerKey = f"{dataset}_{imageId}"

        if centerKey in centers:
            return tuple(centers[centerKey])

        if dataset == "training" and imageId in centers:
            return tuple(centers[imageId])

    center, _ = detectOpticDisc(image, fovMask)
    return center


"""
    align a DRIVE image using the optic disc center
    returns:
        alignedImage, alignedVessels, alignedFov
"""
def alignDriveImage(image, vesselMask, fovMask, discCenter,
                    targetCenter=None):
    height, width = image.shape[:2]

    if targetCenter is None:
        targetCenter = (int(width * 0.80), height // 2)

    centerX, centerY = discCenter

    if centerX < width / 2:
        image = cv2.flip(image, 1)
        vesselMask = cv2.flip(vesselMask, 1)
        fovMask = cv2.flip(fovMask, 1)

        centerX = width - 1 - centerX

    dx = targetCenter[0] - centerX
    dy = targetCenter[1] - centerY

    transform = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])

    alignedImage = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    alignedVessels = cv2.warpAffine(
        vesselMask,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    alignedFov = cv2.warpAffine(
        fovMask,
        transform,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return alignedImage, alignedVessels, alignedFov
