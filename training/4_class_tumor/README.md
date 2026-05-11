Tumor Histopathology 4-Class Classification Model Training (ViT)
This project uses a pre-trained Vision Transformer (ViT) model to fine-tune a 4-class classification task for tumor pathology slides. The codebase includes the complete pipeline from data loading and augmentation to model building and evaluation (supporting both patch-level and patient-level metrics).

File Structure
Please ensure your project directory looks like this, saving the two pieces of code you provided into their respective Python files:

Plaintext
project_folder/
│
├── train.py                # Main training script (model definition, training loop, etc.)
├── utlis_4_class.py        # Utility script (Dataset, data augmentations, evaluation functions, etc.)
└── README.md               # This documentation file
Environment Dependencies
Ensure the following Python libraries are installed before running the code (Python 3.8+ and CUDA-enabled PyTorch are recommended):

Bash
pip install torch torchvision
pip install timm pandas numpy Pillow scikit-learn matplotlib seaborn
Dataset Preparation
The code relies on CSV files to load the data. Please prepare train.csv and val.csv, and ensure they contain the following 3 required columns:

file_path: The absolute or relative local path to the image (e.g., D:/data/image_001.jpg).

label: The class label of the image (integer, 0, 1, 2, 3).

patient_id: A unique identifier for the patient (used to aggregate patient-level metrics).

Example (train.csv):

Code snippet
file_path,label,patient_id
D:/data/train/img1.jpg,0,patient_A
D:/data/train/img2.jpg,0,patient_A
D:/data/train/img3.jpg,1,patient_B
Training Steps
1. Modify Data Paths
Open train.py, locate the # Config section, and modify the CSV paths to your actual local paths:

Python
train_csv = r"Your_absolute_path_to_train.csv"
val_csv   = r"Your_absolute_path_to_val.csv"
2. Start Training
Navigate to the project directory in your terminal and run the following command:

Bash
python train.py
3. Training Process and Outputs
The code defaults to training for 30 Epochs.

Automatic Mixed Precision (AMP, bfloat16) is used during training to accelerate computation and save VRAM.

After each epoch, the terminal will print both Patch-level and Patient-level Accuracy and F1 Score.

Early Stopping: If the patch-level F1 Score does not improve for 3 consecutive epochs, the training will stop early.

Model Saving: The model weights with the best validation performance (Patch-level F1) will be automatically saved in the current directory as best_model_tumor_4class.pth.

Model Details (Advanced)
If you need to modify the model, you can adjust the following strategies in train.py:

Base Model: Uses the kaiko_ai_towards_large_pathology_fms pre-trained model, which is specifically optimized for pathology.

Layer Freezing Strategy: By default, the bottom feature extraction layers are frozen. Only the classification head, the last 8 Transformer Blocks, and the final LayerNorm are unfrozen and fine-tuned.

Optimizer: AdamW combined with a CosineAnnealingLR scheduler. Different learning rates are applied to different layers (5e-5 for the classification head, 1e-5 for the backbone).
