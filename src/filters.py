
import cv2

"""
   applies gaussian denoising to an image
   args:
       image: input image as a numpy array
       kernel_size: size of the gaussian kernel (must be odd)
       sigma: standard deviation
   returns:
       filtered image
"""
def gaussianFilter(image, kernelSize=5, sigma=1.0):
    return cv2.GaussianBlur(
        image,
        (kernelSize, kernelSize),
        sigmaX=sigma
    )

"""
    applies bilateral denoising to an image
    args:
        image: input image as a numpy array
        diameter: diameter of the pixel neighborhood used for filtering
        sigma_color: controls how much intensity/color differences
                     influence the filtering process.
        sigma_space: controls the influence of spatial distance
    returns:
        filtered image with reduced noise and preserved edges
"""
def bilateralFilter(image, diameter=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(
        image,
        d=diameter,
        sigmaColor=sigmaColor,
        sigmaSpace=sigmaSpace
    )

"""
    applies non local means denoising to an rgb image
    (reduces noise by comparing similar image patches)
    args:
        image: input rgb image as a numpy array
        h: filtering strength
        template_window: size of the patches used for similarity comparison (must be odd)
        search_window: size of the area where similar patches are searched (must be odd)
    returns:
        denoised rgb image
"""
def nonLocalMeansFilter(image, h=10, template_window=7, search_window=21):
    # convert rgb to bgr (opencv expects bgr)
    bgrImage = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    filtered = cv2.fastNlMeansDenoisingColored(
        bgrImage,
        None,
        h=h,
        hColor=h,
        templateWindowSize=template_window,
        searchWindowSize=search_window
    )
    # convert back to rgb
    return cv2.cvtColor(filtered, cv2.COLOR_BGR2RGB)
