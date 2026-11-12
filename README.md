# Medical Image Denoising with Anatomical Structure Preservation

This research project aims to develop an adaptive noise reduction method for medical images while preserving fine anatomical structures, such as retinal blood vessels.

The proposed approach uses statistical probability maps generated from annotated medical images to guide the denoising process. Areas with a higher probability of containing important anatomical structures will receive less aggressive filtering.

## Datasets

### DRIVE (Digital Retinal Images for Vessel Extraction)

The primary dataset used for development and initial experiments.
- 40 retinal fundus images
- 20 training and 20 test images
- Manual vessel annotations available for training
- Field-of-view masks

Dataset: https://drive.grand-challenge.org/

### CHASE_DB1

Planned for additional validation to evaluate the method on a different collection of retinal images.

Dataset: https://researchinnovation.kingston.ac.uk/en/datasets/chasedb1-retinal-vessel-reference-dataset-4/

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd <repository-folder>

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Download the DRIVE dataset and place it in `data/DRIVE/`.

Run the current visualization test (in main.py).