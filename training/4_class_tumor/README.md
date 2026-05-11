# Four-Class Brain Tumor ViT Classification Model Training

This section contains the training scripts for the four class tumor ViT model used in OnSight Pathology.

## File Structure 

## Dataset Preparation

The code uses on CSV files to load the data. Please prepare train.csv and val.csv, and ensure they contain the following 3 required columns:

1. file_path: The absolute or relative local path to the image (e.g., D:/data/image_001.jpg).
2. label: The class label of the image (integer, 0, 1, 2, 3).
3. patient_id: A unique identifier for the patient.

Example (train.csv):

    file_path,label,patient_id
    D:/data/train/img1.jpg,0,patient_A
    D:/data/train/img2.jpg,0,patient_A
    D:/data/train/img3.jpg,1,patient_B

## Training Steps

### 1. Modify Data Paths
Open train.py, locate the # Config section, and modify the CSV paths to your actual local paths:

    train_csv = r"Your_absolute_path_to_train.csv"
    val_csv   = r"Your_absolute_path_to_val.csv"

### 2. Start Training
Navigate to the project directory in your terminal and run the following command:

    python train.py

### 3. Training Process and Outputs
* The code defaults to training for 30 Epochs.
* After each epoch, the terminal will print both Patch-level and Patient-level Accuracy and F1 Score.
* Early Stopping: If the patch-level F1 Score does not improve for 3 consecutive epochs, the training will stop early.
  
## Model Details

If you need to modify the model, you can adjust the following strategies in train.py:
* Base Model: Uses the kaiko_ai_towards_large_pathology_fms pre-trained model.
* Layer Freezing Strategy: By default, the bottom feature extraction layers are frozen. Only the classification head, the last 8 Transformer Blocks are unfrozen and fine-tuned.
