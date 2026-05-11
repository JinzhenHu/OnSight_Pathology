# Magnification Detection Model Training (ViT)

This repo contains the training scripts for the magnification detector in OnSight Pathology. It is used to classify the magnification level of pathology image patches into four categories: 5x, 10x, 20x, and 40x.

## File Structure

Please organize your project directory as follows, saving your two scripts accordingly:

    project_folder/
    │
    ├── train.py                
    ├── utlis.py                
    └── README.md               

## Dataset Preparation

Unlike CSV-based loading, this script reads images directly from a structured directory. 

1. Root Directory: You must define a root directory containing subfolders named exactly after your classes (`5x`, `10x`, `20x`, `40x`).
2. File Naming Rules: The `TumorDataset` class has strict filtering rules for images:
   * The file extension must be `.jpg`, `.jpeg`, or `.png`.
   * The filename MUST contain the string `patch_1`.
   * The filename MUST contain a valid TCGA patient ID matching the regex pattern: `TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}`. This ID is extracted and used to ensure patches from the same patient do not leak across the train and validation sets.

Example Directory Structure:

    OnSight_Pathology/
    ├── 5x/
    │   ├── TCGA-XX-0001_patch_1.jpg
    │   └── TCGA-XX-0002_patch_1.jpg
    ├── 10x/
    │   └── TCGA-XX-0001_patch_1.jpg
    ├── 20x/
    └── 40x/

## Training Steps

### 1. Modify the Data Path
Open `train.py`, locate the `root_path` variable at the beginning of the `main()` function, and update it to point to your local dataset directory:

    root_path = r"Your_absolute_path_to_OnSight_Pathology"

### 2. Run the Script
Navigate to the project directory in your terminal and execute the training script:

    python train.py

### 3. Training Process and Outputs
* Data Splitting: The script automatically shuffles the unique patient IDs and performs a 70% / 30% split for training and validation.
* Early Stopping: The script monitors Validation Accuracy. If the accuracy does not improve for 5 consecutive epochs, training stops early.
* Model Saving: The model weights yielding the highest validation accuracy are saved as `best_model.pth` in the current directory.

## Model Details

If you need to adjust hyperparameters, note the following configurations established in `train.py`:
* Base Model: Uses `vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms`, a small ViT pre-trained on pathology data.
* Custom Classification Head: Replaces the default head with a Multi-Layer Perceptron (MLP) containing a Linear layer (1024 units), GELU activation, Dropout (0.3), and a final Linear projection to the 4 classes.
