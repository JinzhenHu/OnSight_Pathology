import os
import timm
import torch
import random
import numpy as np
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset,DataLoader,Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode, v2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


train_transform = v2.Compose([
    v2.ToImage(),
    #v2.RandomCrop(size=1024),
    v2.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.8, 1.25), interpolation=InterpolationMode.BILINEAR),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomApply([v2.RandomRotation(degrees=(0, 360))], p=0.5),
    v2.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
    v2.ToDtype(torch.float32, scale=True),
    v2.RandomApply([v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
    v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

val_transform = v2.Compose([
    v2.ToImage(),
    #v2.CenterCrop(size=1024),
    v2.Resize(size=224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])




class TumorDataset(Dataset):
    def __init__(self, csv_file, transform=None, max_imgs_per_patient=None, seed=42):
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.max_imgs_per_patient = max_imgs_per_patient

        random.seed(seed)

        # =========================================================
        # 1. split by patient_id
        # =========================================================
        patient_dict = defaultdict(list)

        for _, row in self.df.iterrows():
            pid = row["patient_id"]
            patient_dict[pid].append(row)

        # =========================================================
        # 2. determine max images per patient
        # =========================================================
        if max_imgs_per_patient == "min":
            min_imgs = min(len(v) for v in patient_dict.values())
            print(f"Using min images per patient: {min_imgs}")
            max_imgs_per_patient = min_imgs

        # =========================================================
        # 3. sample images for each patient
        # =========================================================
        self.samples = []

        for pid, items in patient_dict.items():
            if max_imgs_per_patient is None:
                selected = items
            else:
                if len(items) > max_imgs_per_patient:
                    selected = random.sample(items, max_imgs_per_patient)
                else:
                    selected = items

            self.samples.extend(selected)

        # =========================================================
        # 4. flatten to list
        # =========================================================
        self.file_paths = [row["file_path"] for row in self.samples]
        self.labels = [int(row["label"]) for row in self.samples]
        self.patient_id = [row["patient_id"] for row in self.samples]

        print(f"Total samples after selection: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        patient_id = self.patient_id[idx]

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, patient_id
    


def evaluate_model(model, dataloader, criterion, device, label_names, num_classes, plot_cm=False):
    model.eval()

    total_loss = 0.0

    all_patch_labels = []
    all_patch_preds = []
    all_patch_probs = []

    patient_probs = defaultdict(list)
    patient_labels = {}

    with torch.no_grad():
        for images, labels, patient_id in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            total_loss += loss.item()

            outputs = outputs.float()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_patch_labels.extend(labels.detach().cpu().numpy().tolist())
            all_patch_preds.extend(preds.detach().cpu().numpy().tolist())
            all_patch_probs.extend(probs.detach().cpu().numpy())

            for i, pid in enumerate(patient_id):
                pid = str(pid)
                patient_probs[pid].append(probs[i].detach().cpu().numpy())
                patient_labels[pid] = int(labels[i].item())

    avg_loss = total_loss / len(dataloader)

    patch_metrics = compute_metrics(
        y_true=all_patch_labels,
        y_pred=all_patch_preds,
        num_classes=num_classes
    )

    patient_true = []
    patient_pred = []

    for pid, prob_list in patient_probs.items():
        mean_prob = np.mean(prob_list, axis=0)
        pred_label = int(np.argmax(mean_prob))
        true_label = patient_labels[pid]

        patient_true.append(true_label)
        patient_pred.append(pred_label)

    patient_metrics = compute_metrics(
        y_true=patient_true,
        y_pred=patient_pred,
        num_classes=num_classes
    )

    if plot_cm:
        plot_confusion_matrices(
            patch_true=all_patch_labels,
            patch_pred=all_patch_preds,
            patient_true=patient_true,
            patient_pred=patient_pred,
            label_names=label_names
        )

    return avg_loss, patch_metrics, patient_metrics


def compute_metrics(y_true, y_pred, num_classes):
    average_type = "binary" if num_classes == 2 else "macro"

    return {
        "acc": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average_type, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average_type, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=average_type, zero_division=0),
    }


def plot_confusion_matrices(patch_true, patch_pred, patient_true, patient_pred, label_names):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    cm_patch = confusion_matrix(patch_true, patch_pred)
    axes[0].imshow(cm_patch, interpolation="nearest")
    axes[0].set_title("Patch-level Confusion Matrix")
    axes[0].set_xticks(range(len(label_names)))
    axes[0].set_yticks(range(len(label_names)))
    axes[0].set_xticklabels(label_names, rotation=45, ha="right")
    axes[0].set_yticklabels(label_names)
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    for i in range(cm_patch.shape[0]):
        for j in range(cm_patch.shape[1]):
            axes[0].text(j, i, str(cm_patch[i, j]), ha="center", va="center")

    cm_patient = confusion_matrix(patient_true, patient_pred)
    axes[1].imshow(cm_patient, interpolation="nearest")
    axes[1].set_title("Patient-level Confusion Matrix")
    axes[1].set_xticks(range(len(label_names)))
    axes[1].set_yticks(range(len(label_names)))
    axes[1].set_xticklabels(label_names, rotation=45, ha="right")
    axes[1].set_yticklabels(label_names)
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    for i in range(cm_patient.shape[0]):
        for j in range(cm_patient.shape[1]):
            axes[1].text(j, i, str(cm_patient[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.show()