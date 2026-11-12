
import numpy as np

"""
    add noise to an image dor testing filters later on
    args:
        image: input image as a numpy array 
        sigma: standard deviation 
        seed: random seed for experiments
    returns:
        noisy image as a numpy array 
"""
def addNoise(image, sigma=20, seed=42):
    rng = np.random.default_rng(seed)

    noise = rng.normal(0, sigma, image.shape)

    noisyImage = image.astype(np.float32) + noise

    noisyImage = np.clip(noisyImage, 0, 255)

    return noisyImage.astype(np.uint8)
