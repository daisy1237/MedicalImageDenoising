
import cv2
import numpy as np


def classifyVessels(vesselMask, threshold=2.0):
    vessels = (vesselMask > 0).astype(np.uint8)

    distance = cv2.distanceTransform(
        vessels,
        cv2.DIST_L2,
        5
    )

    skeleton = np.zeros_like(vessels)

    remaining = vessels.copy()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_CROSS, (3, 3)
    )

    while np.any(remaining):
        eroded = cv2.erode(remaining, kernel)
        opened = cv2.dilate(eroded, kernel)

        edges = cv2.subtract(
            remaining, opened
        )

        skeleton = cv2.bitwise_or(
            skeleton, edges
        )

        remaining = eroded

    thinSkeleton = (
        (skeleton > 0) &
        (distance <= threshold)
    )

    thickSkeleton = (
        (skeleton > 0) &
        (distance > threshold)
    )

    return thinSkeleton, thickSkeleton
