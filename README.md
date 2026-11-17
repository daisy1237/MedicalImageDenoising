# Anatomy-Guided Adaptive Denoising for OCT Images

This project explores a way to reduce noise in medical OCT images while preserving important anatomical details.

The main idea is to use a **probability map of a retinal layer** to decide where stronger or weaker filtering should be applied.

The project is implemented in Python using traditional image processing techniques, without neural networks.

## 1. Dataset

The project uses the **Duke OCT DME dataset (2015_BOE_Chiu)**, which contains retinal OCT images and manually annotated retinal layers.

We worked with 10 subjects, each having 11 annotated slices.

The subjects were divided as follows:

| Subjects | Purpose |
|---|---|
| 01–06 | Building the anatomical probability map |
| 07–08 | Testing and choosing parameters |
| 09–10 | Final evaluation |

The dataset is stored locally in `data/` and is not included in the repository.

## 2. How It Works

### Anatomical Probability Map

We focused on the **Outer Plexiform Layer (OPL)** of the retina.

Since retinal layers have different shapes and positions across subjects, we first normalize the OCT images using two anatomical boundaries.

We then combine the annotated OPL regions from multiple subjects to create a probability map.

This map estimates how likely each pixel is to belong to the OPL region.

### Adaptive Filtering

We implemented several versions of an adaptive filtering method.

The main approach combines two Gaussian filters:

- **Weak Gaussian:** preserves more local details but removes less noise.
- **Strong Gaussian:** removes more noise but can blur details.

The probability map controls how much each filter contributes to the final image.

The final version, **Adaptive V4**, also considers the noise level. When noise increases, the method gives more importance to stronger filtering.

## 3. Experiments

We tested the methods by adding synthetic Gaussian noise to the OCT images at three levels:

- Low noise: `σ = 0.04`
- Medium noise: `σ = 0.08`
- High noise: `σ = 0.12`

We compared:

- Weak Gaussian
- Strong Gaussian
- Adaptive V2
- Adaptive V4
- Noise-Aware Control (adaptive filtering without the spatial probability map)

The main evaluation metrics were PSNR, MAE and Gradient Error, calculated globally or in the OPL region.

## 4. Results

The final evaluation used 22 annotated OCT slices from subjects 09 and 10.

### OPL PSNR (dB)

Higher values are better.

| Method | Low noise | Medium noise | High noise |
|---|---:|---:|---:|
| Weak Gaussian | 33.4123 | 27.8092 | 24.4060 |
| Strong Gaussian | 32.2433 | 31.2385 | 29.9749 |
| Adaptive V2 | 33.5846 | 30.7743 | 28.2995 |
| **Adaptive V4** | **33.5846** | **31.4267** | 29.9825 |
| Noise-Aware Control | 32.7600 | 31.3244 | **29.9828** |

### What We Observed

- At **low noise**, Adaptive V4 performed best in the OPL region, together with V2.
- At **medium noise**, Adaptive V4 achieved the best OPL PSNR and Gradient Error.
- At **high noise**, Adaptive V4 performed almost identically to Strong Gaussian and Noise-Aware Control.
- The anatomical probability map was most useful at low and medium noise levels.

We also generated visual comparisons to observe the differences between the filtering methods.

## 5. Project Structure

```text
tema4/
├── data/                   # OCT dataset (not tracked)
├── src/
│   ├── filters.py
│   ├── metrics.py
│   ├── retinalNormalization.py
│   ├── adaptiveFilter.py
│   └── viewOCT.py
├── experiments/            # Probability maps and evaluations
├── results/
│   ├── final_figures/
│   ├── visual_comparisons/
│   └── *.csv
├── main.py
└── README.md
```

## 6. Technologies

- Python
- NumPy
- OpenCV
- SciPy
- scikit-image
- Matplotlib
- pandas

## 7. Limitations

There are a few limitations to the current approach:

- The dataset is relatively small.
- The original OCT images already contain some noise.
- We tested synthetic Gaussian noise, which does not fully represent real OCT noise.
- The normalization process uses manually annotated retinal boundaries.
- The noise level is known during testing rather than estimated automatically.
- The probability map represents the OPL region, not individual microscopic structures.

## 8. Conclusion

We developed an adaptive denoising method that uses anatomical information and the noise level to adjust filtering strength.

**Adaptive V4 showed better results in the OPL region at low and medium noise levels**, compared with filtering without the anatomical probability map.

At high noise levels, its performance was very similar to strong Gaussian filtering.

Overall, the experiments suggest that anatomical probability maps can be useful for guiding image denoising, although more testing would be needed to confirm how well the method preserves individual fine anatomical structures.