import os
import timm
import torch
import numpy as np
from PIL import Image
import seaborn as sns
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode, v2
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from torch.utils.data import Dataset
from pathlib import Path
import pandas as pd
import torch.nn.functional as F
import re
import random
    
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomCrop(224),

    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(180),
    v2.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.05),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1,1.0)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.CenterCrop(size=224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5,0.5,0.5),std=(0.5,0.5,0.5)),
])


class TumorDataset(Dataset):
    def __init__(self, image_root, label_mapping, transform=None,max_cases=None):
        self.images = []
        self.labels = []
        self.slide_ids = []
        self.transform = transform
        self.label_mapping = label_mapping

        for tumor_type, label in self.label_mapping.items():
            class_path = os.path.join(image_root, tumor_type)

            patient_imgs = {}

            for img_name in os.listdir(class_path):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')) and "patch_1" in img_name:

                    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", img_name)

                    patient_id = match.group(1)

                    if patient_id not in patient_imgs:
                        patient_imgs[patient_id] = os.path.join(class_path, img_name)

            for patient_id, img_path in patient_imgs.items():
                self.images.append(img_path)
                self.labels.append(label)
                self.slide_ids.append(patient_id)
        
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        if self.transform: image = self.transform(image)
        return image,self.labels[idx],self.slide_ids[idx]

    def __len__(self):
        return len(self.images)
    
class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset   
        self.transform = transform

    def __getitem__(self, index):
        x,y,slide_ids = self.subset[index]
        if self.transform: x = self.transform(x)
        return x,y,slide_ids

    def __len__(self):
        return len(self.subset)



def evaluate_patch_level(model, dataloader, device, label_names, plot_cm=True):
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print("\n evaluating patch-level performance...")
    
    with torch.no_grad():
        for images, labels, _ in dataloader:  
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n" + "Patch-level metrics".center(50, "="))
    print(f"Overall Accuracy: {acc:.4f}")
    print(f"Macro F1-Score:   {f1_macro:.4f}")
    print(f"Weighted F1-Score:{f1_weighted:.4f}")
    print("-" * 50)
    
    labels = list(range(len(label_names)))

    report = classification_report(
        all_labels,
        all_preds,
        labels=labels,
        target_names=label_names,
        digits=4,
        zero_division=0
    )
    print(report)
    if plot_cm:
        cm = confusion_matrix(all_labels, all_preds)
        cm_df = pd.DataFrame(cm, index=[f"True {n}" for n in label_names], 
                                columns=[f"Pred {n}" for n in label_names])
        
        print("\n[ confusion matrix]")
        print(cm_df)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_names, yticklabels=label_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Patch-level Confusion Matrix')
        plt.show()
    

    return all_labels, all_preds
