
import SimpleITK as sitk

"""
    resample a 2D or 3D image to a reference image 
    args:
        image: source simple itk image.
        reference: image defining the target grid
        isMask: whether the source is a segmentation mask
    returns:
        resampled simple itk image
"""
def resampleToReference(image, reference, isMask=False):
    if image.GetDimension() != reference.GetDimension():
        raise ValueError(
            "source and reference must have the same dimension"
        )
    interpolator = (
        sitk.sitkNearestNeighbor
        if isMask
        else sitk.sitkLinear
    )

    return sitk.Resample(
        image,
        reference,
        sitk.Transform(image.GetDimension(), sitk.sitkIdentity),
        interpolator,
        0.0,
        image.GetPixelID()
    )

"""
    resample segmentation masks to a common reference grid
    args:
        masks: list of simple itk segmentation images
        reference: reference image defining the target grid
    returns:
        list of normalized binary numpy arrays
"""
def normalizeMasks(masks, reference):
    normalized = []

    for mask in masks:
        resampled = resampleToReference(mask, reference, isMask=True)

        normalized.append(
            sitk.GetArrayFromImage(resampled) > 0
        )

    return normalized

"""
    resample a probability map from its reference grid into the physical grid of a target image
    args:
        probabilityMap: numpy array containing probabilities
        reference: simple itk image defining the map's grid
        target: target simple itk image
    returns:
        probability map resampled to the target grid
"""
def probabilityMapToImage(probabilityMap, reference, target):
    import numpy as np

    if probabilityMap.shape != sitk.GetArrayFromImage(reference).shape:
        raise ValueError(
            "probability map shape does not match its reference"
        )

    mapImage = sitk.GetImageFromArray(
        probabilityMap.astype(np.float32)
    )

    mapImage.CopyInformation(reference)

    resampled = resampleToReference(mapImage, target, isMask=False)

    return np.clip(
        sitk.GetArrayFromImage(resampled),
        0.0,
        1.0
    )
