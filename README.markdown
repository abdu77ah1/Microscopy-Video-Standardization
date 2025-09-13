# Microscopy Video Standardization

This repository contains the implementation of the methods described in the paper:

**Automated Preprocessing and Standardization of Microscopy Videos for Studying Cell Migration**  
Abdullah Mahmood, Sophie Féréol, Régis Fournier, Amine Nait-Ali

The goal of this project is to automate the preprocessing of time-lapse microscopy videos of migrating cells, ensuring homogeneity, stability, and improved contrast prior to downstream analysis of cell migration trajectories.

## Features

The preprocessing pipeline performs the following steps:

### Bit Depth Conversion
- Converts raw 16-bit TIFF stacks into 8-bit PNG frames.
- Normalizes pixel intensity values into a 0–255 range for compatibility and consistency.

### Video Stabilization
- Uses SIFT-based keypoint detection to align frames and correct stage drifts.
- Reduces jitter and movement artifacts caused by microscope instability.

### Cropping
- Removes artifacts and black borders introduced during stabilization by cropping 10 pixels from each edge.

### Illumination Normalization
- Corrects brightness fluctuations across frames.
- Matches each frame’s brightness to the global average, ensuring consistent illumination.

### Quality Metrics
We evaluate the effectiveness of preprocessing using PSNR and SSIM:
- **PSNR (Peak Signal-to-Noise Ratio)**: > 32 dB across videos, ensuring minimal distortion.
- **SSIM (Structural Similarity Index)**: ≈ 0.98, confirming preservation of structural information.

In addition, contrast enhancement was quantified by measuring the standard deviation of pixel intensities, which increased after illumination normalization, reflecting improved frame clarity.

## Results
- **Consistency**: Processed videos are stable, well-contrasted, and uniform.
- **Efficiency**: Automated preprocessing reduces runtime by 91% compared to manual ImageJ-based methods (≈45 minutes vs. 500 minutes).
- **Accessibility**: Converts TIFF microscopy videos into a format usable in standard image viewers and video tools.

## Installation

Clone this repository:
```bash
git clone https://github.com/abdu77ah1/Microscopy-Video-Standardization.git
cd Microscopy-Video-Standardization
```

Create a conda environment and install dependencies:
```bash
conda create -n microproc python=3.9
conda activate microproc
pip install -r requirements.txt
```

## Usage

Run the preprocessing script on a folder of TIFF stacks:
```bash
python preprocess.py --input ./data/raw_tiffs --output ./data/processed_videos
```

### Arguments
- `--input`: Path to the directory containing TIFF stack videos.
- `--output`: Directory where processed videos will be saved.

The script will automatically:
- Convert TIFF stacks into 8-bit frames.
- Apply stabilization.
- Crop artifacts.
- Normalize illumination.
- Recombine frames into standardized videos.

## Citation

If you use this repository, please cite the paper:

```bibtex
@inproceedings{mahmood2025microscopy,
  title={Automated Preprocessing and Standardization of Microscopy Videos for Studying Cell Migration},
  author={Mahmood, Abdullah and Féréol, Sophie and Fournier, Régis and Nait-Ali, Amine},
  booktitle={IEEE Conference},
  year={2025}
}
```

## Acknowledgments

This work was supported by:
- INSERM, U955 IMRB, BNMS Team
- LISSI Laboratory, Université Paris-Est Créteil
- Association Française contre les Myopathies (AFM) – TRANSLAMUSCLE Projects #19507 and #22946
- Erasmus Mundus Joint Master’s in Photonics for Security, Reliability, and Safety (PSRS)