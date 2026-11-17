
import numpy as np


"""
    flatten an OCT image using a retinal reference boundary
    args:
        image: 2D OCT image
        layers: manual retinal boundaries (8, width)
        referenceIndex: boundary index (0-based)
        targetY: target vertical position
    returns:
        flattened image, vertical shifts, valid reference columns
"""
def flattenOCT(image, layers, referenceIndex=7, targetY=None):
    height, width = image.shape

    reference = layers[referenceIndex]
    validReference = (
        np.isfinite(reference)
        & (reference >= 1)
        & (reference <= height)
    )

    if not np.any(validReference):
        raise ValueError("No valid reference boundary found")

    if targetY is None:
        targetY = height // 2

    # Estimate shifts only where the reference is available.
    # Missing columns are not used as anatomical observations.
    shifts = np.zeros(width, dtype=np.int32)

    shifts[validReference] = np.rint(
        targetY - (reference[validReference] - 1)
    ).astype(np.int32)

    flattened = np.zeros_like(image)

    for x in range(width):
        if not validReference[x]:
            continue

        shift = shifts[x]

        sourceStart = max(0, -shift)
        sourceEnd = min(height, height - shift)

        targetStart = sourceStart + shift
        targetEnd = sourceEnd + shift

        if sourceStart < sourceEnd:
            flattened[targetStart:targetEnd, x] = (
                image[sourceStart:sourceEnd, x]
            )

    return flattened, shifts, validReference


"""
    apply the same vertical shifts to a binary mask
"""
def flattenMask(mask, shifts, validReference):
    height, width = mask.shape

    flattened = np.zeros_like(mask)

    for x in range(width):
        if not validReference[x]:
            continue

        shift = int(shifts[x])

        sourceStart = max(0, -shift)
        sourceEnd = min(height, height - shift)

        targetStart = sourceStart + shift
        targetEnd = sourceEnd + shift

        if sourceStart < sourceEnd:
            flattened[targetStart:targetEnd, x] = (
                mask[sourceStart:sourceEnd, x]
            )

    return flattened


from scipy.ndimage import map_coordinates


"""
    normalize an OCT image between two anatomical boundaries
    args:
        image: original OCT image (height, width)
        layers: manual boundaries (8, width)
        upperIndex: upper reference boundary
        lowerIndex: lower reference boundary
        targetTop: normalized upper position
        targetBottom: normalized lower position
    returns:
        normalized image, source coordinates,
        valid anatomical support
"""
def normalizeRetinaTwoBoundaries(
        image, layers,
        upperIndex=0, lowerIndex=7,
        targetTop=100, targetBottom=350):

    height, width = image.shape

    if not (0 <= targetTop < targetBottom < height):
        raise ValueError("Invalid target interval")

    upper = layers[upperIndex]
    lower = layers[lowerIndex]

    validColumns = (
        np.isfinite(upper)
        & np.isfinite(lower)
        & (upper >= 1)
        & (lower <= height)
        & (upper < lower)
    )

    sourceY = np.zeros((height, width), dtype=np.float32)
    support = np.zeros((height, width), dtype=bool)

    targetRows = np.arange(
        targetTop, targetBottom + 1
    )

    relativeY = (
        (targetRows - targetTop)
        / (targetBottom - targetTop)
    )

    for x in np.where(validColumns)[0]:
        # MATLAB coordinates are 1-based
        yUpper = upper[x] - 1
        yLower = lower[x] - 1

        sourceY[targetRows, x] = (
            yUpper + relativeY * (yLower - yUpper)
        )

        support[targetRows, x] = True

    sourceX = np.broadcast_to(
        np.arange(width, dtype=np.float32)[None, :],
        (height, width)
    )

    coordinates = np.stack(
        [sourceY, sourceX]
    )

    normalizedImage = map_coordinates(
        image.astype(np.float32),
        coordinates,
        order=1,
        mode="constant",
        cval=0
    )

    normalizedImage[~support] = 0

    return normalizedImage, coordinates, support


"""
    transform a binary mask using the same
    anatomical coordinate mapping
"""
def normalizeMaskTwoBoundaries(mask, coordinates, support):
    normalized = map_coordinates(
        mask.astype(np.uint8),
        coordinates,
        order=0,
        mode="constant",
        cval=0
    )

    normalized[~support] = 0

    return normalized.astype(np.uint8)
