# MedSAM2 with MedYOLO and RECIST for 3D Lesion Segmentation

This project provides a framework for 3D medical image segmentation using the MedSAM2 model. It implements and compares two distinct prompting strategies to guide the segmentation process:

1.  **RECIST-based Prompting**: Utilizes RECIST (Response Evaluation Criteria in Solid Tumors) markers, simulated from ground truth data, as precise point-based prompts for MedSAM2.
2.  **MedYOLO-based Prompting**: Employs a MedYOLO model to automatically detect 3D bounding boxes of lesions. These bounding boxes are then converted into corner-point prompts to guide MedSAM2, creating a fully automated segmentation pipeline.

The main inference script, `medsam2_infer_3D_CT.py`, runs both pipelines and saves the segmentation results, evaluation metrics (DSC and NSD), and visualizations for comparison.

## Setup

### 1. Environment Setup

It is recommended to create a virtual environment to manage dependencies.

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

Install the required Python packages for the main MedSAM2 project and the MedYOLO submodule.

```bash
pip install -r requirements.txt # Or install from pyproject.toml if available
pip install -r MedYOLO/requirements.txt
```

### 3. Download Checkpoints

The project requires two sets of checkpoints: one for MedSAM2 and one for MedYOLO.

**MedSAM2 Checkpoint:**

Run the provided script to download the main MedSAM2 checkpoint. This will place the `checkpoint.pt` file in the `FLARE_results/checkpoints/` directory.

```bash
sh download.sh
```

**MedYOLO Checkpoint:**

The checkpoint for the MedYOLO model is already included in this repository at:

```
MedYOLO/runs/train/exp3/weights/last.pt
```

No download is necessary for the MedYOLO checkpoint.

## Usage

The primary script for running inference is `medsam2_infer_3D_CT.py`.

### Data Preparation

The training and validation data used in this project can be downloaded from the [FLARE-Task1-PancancerRECIST-to-3D dataset on Hugging Face](https://huggingface.co/datasets/FLARE-MedFM/FLARE-Task1-PancancerRECIST-to-3D/tree/main).

Place your input data, in `.npz` format, into the `data/validation_npz/` directory. The script will process all `.npz` files found there by default. You can specify a different directory using the `--imgs_path` argument.

### Running Inference

You can run inference using either the default RECIST-based strategy or the automated MedYOLO-based strategy. The script will run both by default for comparison if the `--use-medyolo` flag is enabled.

**RECIST-based Segmentation (Default):**

This command runs the segmentation pipeline using ground-truth-derived RECIST markers as prompts.

```bash
python medsam2_infer_3D_CT.py \
    --cfg configs/sam2.1_hiera_t512.yaml \
    --imgs_path data/validation_npz \
    --pred_save_dir FLARE_results
```

**MedYOLO-based Segmentation:**

Add the `--use-medyolo` flag to enable the MedYOLO pipeline. The script will first detect bounding boxes and then use them as prompts for MedSAM2.

```bash
python medsam2_infer_3D_CT.py \
    --cfg configs/sam2.1_hiera_t512.yaml \
    --use-medyolo \
    --imgs_path data/validation_npz \
    --pred_save_dir FLARE_results
```

### Key Command-Line Arguments

-   `--checkpoint`: Path to the MedSAM2 model checkpoint. (Default: `FLARE_results/checkpoints/checkpoint.pt`)
-   `--imgs_path`: Path to the directory containing input `.npz` files. (Default: `data/validation_npz`)
-   `--pred_save_dir`: Directory where segmentation results, visualizations, and metrics will be saved. (Default: `FLARE_results`)
-   `--cfg`: Path to the model config file. (Default: `configs/sam2.1_hiera_t512.yaml`)
-   `--use-medyolo`: A flag to enable the MedYOLO-based prompting strategy.
-   `--medyolo-weights`: Path to the MedYOLO model weights. (Default: `MedYOLO/runs/train/exp3/weights/last.pt`)

## Results

The script generates the following outputs in the specified prediction directory (`FLARE_results` by default):

-   **Segmentation Masks**: 3D segmentation masks are saved as NIfTI files (`.nii.gz`), with separate files for each strategy (e.g., `*_mask_recist.nii.gz` and `*_mask_medyolo.nii.gz`).
-   **Visualizations**: PNG images are created for each case, showing a key slice with the model's segmentation overlay and the prompts used (e.g., `*_recist_visualization.png` and `*_medyolo_visualization.png`).
-   **Evaluation Metrics**: Detailed metrics, including Dice Similarity Coefficient (DSC) and Normalized Surface Distance (NSD), are saved in `.csv` files (`evaluation_metrics_recist.csv` and `evaluation_metrics_medyolo.csv`). A summary is also printed to the console.
