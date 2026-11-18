# Medical Image Denoising with Anatomical Probability Maps

## Overview

This project explores a method for reducing noise in medical images while preserving fine anatomical structures.

The main idea is to build an anatomical probability map from annotated images and use it to control the strength of denoising. Regions with a higher probability of containing important structures receive less smoothing.

The implementation uses Python, OpenCV and NumPy, without neural networks.

## Datasets

Two retinal imaging datasets were used:

- **Duke OCT:** Retinal OCT images with annotated layers, used to investigate the preservation of the Outer Plexiform Layer (OPL).
- **DRIVE:** Retinal fundus images, used to investigate the preservation of blood vessels. RITE vessel annotations were used for test evaluation.

Training and test images are kept separate when constructing the anatomical probability maps.

## Method

The main steps are:

1. Load and preprocess the medical images.
2. Align anatomical structures across training images.
3. Build an anatomical probability map from the annotations.
4. Apply adaptive Gaussian filtering, using weaker smoothing in regions with higher anatomical probability.
5. Evaluate the results against standard Gaussian filtering and a noise-aware control method.

For DRIVE, the final adaptive method uses a smoothed probability map with a weighting factor of `alpha = 0.5`.

## Results

### OCT

Experiments on the Duke OCT dataset showed that anatomical probability maps can help preserve the OPL during denoising, particularly at lower noise levels.

### DRIVE

The final evaluation used 20 DRIVE test images.

| Metric | Adaptive Tuned | Noise-Aware Control | Strong Gaussian |
|---|---:|---:|---:|
| Global PSNR (dB) | 32.9566 | 32.9226 | 32.9458 |
| Vessel PSNR (dB) | 32.7241 | 32.6839 | 32.6314 |
| Thin-vessel centerline PSNR (dB) | 32.7604 | 32.7089 | 32.6554 |
| Thick-vessel centerline PSNR (dB) | 32.5663 | 32.5449 | 32.4301 |

The adaptive method achieved small average improvements, especially for thin-vessel centerline reconstruction.

The results are exploratory because the final weighting factor was selected using DRIVE test results.

## Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- Pandas
- scikit-image

## Running the Experiments

Install the required Python dependencies and run the experiments from the project root.

For the final DRIVE evaluation:

```bash
python -m experiments.evaluateDriveAdaptive
```

To generate the final figures:

```bash
python -m experiments.generateDriveFinalFigures
```

Results and figures are saved in the `results/` directory.

## Conclusion

The experiments suggest that anatomical probability maps can improve the preservation of fine retinal structures during image denoising.

The improvements are relatively small, and further validation on independent datasets and