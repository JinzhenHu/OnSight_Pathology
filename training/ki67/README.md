# Ki67 Model – Training Pipeline

This directory contains the full data preparation and training pipeline for the Ki67 detection model.

The model was trained using annotations generated in QuPath and exported as GeoJSON files.

The pipeline consists of two major stages:

1. Data preparation (`setup_pipeline/`)
2. Model training (`train/`)

---

# Data Preparation

All preprocessing scripts are located in:

```
setup_pipeline/
```

The workflow converts QuPath GeoJSON annotations into COCO format, optionally performs augmentation, prepares data for Ultralytics YOLO training, and creates train/test splits.

The steps must be executed in order.

---

## Step 1 – Convert QuPath Annotations to COCO

Script:

```
convert_to_coco_step1.py
```

### Input:
- Whole-slide image (WSI)
- Corresponding QuPath GeoJSON annotation file

### Output:
- Extracted image tiles (JPG)
- COCO-format annotation file
- Output directory containing tiles and annotations (named `out/`)

This script:
- Reads the GeoJSON annotation regions
- Extracts the specified regions from the WSI
- Saves each region as a tile
- Generates COCO-formatted annotations aligned with the extracted tiles

After this step, you will have:
- A folder containing tile patches
- A corresponding COCO annotation file

---

## Step 1.5 – Optional Data Augmentation

Script:

```
create_augmented_training_examples_step1.5.py
```

This step is optional.

### Procedure:

1. Create a separate folder.
2. Copy selected tile patches and the associated COCO annotation file into this folder.
3. Run the augmentation script.

### Output:
- Augmented tile patches
- COCO annotations (structure unchanged)

If you choose to use augmented examples for training:
- Copy the augmented images back into the original `out/` directory created in Step 1.

---

## Step 2 – Convert to Ultralytics Format

Script:

```
to_ultralytics_step2.py
```

This script converts the COCO dataset into the format required for Ultralytics YOLO training.

Output:
- YOLO-formatted labels
- Dataset structure compatible with Ultralytics

---

## Step 3 – Create Train/Test Split

Script:

```
make_train_test_step3.py
```

This script:
- Splits the dataset into training and test sets
- Creates the final directory structure required for training

After this step, the dataset is ready for training.

---

# Model Training

Training scripts are located in:

```
train/
```

Contents:

- `train.py`
- `train.yaml`

---

## Configuration

Before training, edit:

```
train.yaml
```

Specify the path to the dataset created in Step 3:

```yaml
path: /path/to/train_test_dataset
```

Ensure that:
- The training folder structure matches Ultralytics expectations
- The paths are absolute or correctly resolved relative paths

---

## Run Training

From within the `train/` directory:

```bash
python train.py
```

This will initiate model training using the configuration defined in `train.yaml`.

---

# Pipeline Summary

1. `convert_to_coco_step1.py`
2. (Optional) `create_augmented_training_examples_step1.5.py`
3. `to_ultralytics_step2.py`
4. `make_train_test_step3.py`
5. Edit `train.yaml`
6. Run `train.py`

---

# Notes

- Annotations originate from QuPath and are exported in GeoJSON format.
- The conversion pipeline ensures alignment between WSI-extracted tiles and annotations.
- The final dataset is formatted for Ultralytics YOLO training.
- This pipeline assumes familiarity with WSI handling and object detection workflows.
