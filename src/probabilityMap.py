
import numpy as np

"""
    build a statistical probability map from binary masks
    (supports both 2d images and 3d volumes)
    args:
        masks: list of binary segmentation masks
               (all masks must have the same shape and be aligned)
        validMasks: optional list of binary masks indicating valid anatomical regions for each sample
    returns:
        probability map as a numpy array with values in [0, 1]
"""
def buildProbabilityMap(masks, validMasks=None):
    if not masks:
        raise ValueError("at least one segmentation mask is required")

    referenceShape = np.asarray(masks[0]).shape

    if len(referenceShape) not in (2, 3):
        raise ValueError("only 2D and 3D masks are supported")

    if validMasks is not None and len(validMasks) != len(masks):
        raise ValueError(
            "the number of valid masks must match the number of masks"
        )

    structureCount = np.zeros(referenceShape, dtype=np.float64)
    validCount = np.zeros(referenceShape, dtype=np.float64)

    for index, mask in enumerate(masks):
        mask = np.asarray(mask)
        if mask.shape != referenceShape:
            raise ValueError(
                "all masks must have the same shape and spatial alignment"
            )
        binary_mask = mask > 0

        if validMasks is None:
            validRegion = np.ones(referenceShape, dtype=bool)
        else:
            validRegion = np.asarray(validMasks[index]) > 0

            if validRegion.shape != referenceShape:
                raise ValueError(
                    "valid region mask has an incompatible shape"
                )

        structureCount += binary_mask & validRegion
        validCount += validRegion

    probabilityMap = np.divide(
        structureCount,
        validCount,
        out=np.zeros_like(structureCount),
        where=validCount > 0
    )

    return probabilityMap.astype(np.float32)
