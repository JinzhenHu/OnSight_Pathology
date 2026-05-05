import os
import timm
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from utlis_4_class import train_transform, val_transform, evaluate_model, TumorDataset



def main():

    # =========================
    # Config
    # =========================
    train_csv = r"D:\JHU\Dimmandias_lab\VIT\train\four_class_split_csv\train.csv"
    val_csv   = r"D:\JHU\Dimmandias_lab\VIT\train\four_class_split_csv\val.csv"

    label_names =    ["Epithelial pattern",
    "Glial histology",
    "Meningothelial Histology",
    "Schwannian histology"]
    num_classes = 4

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    image_num = None

    # =========================
    # Dataset
    # =========================
    train_data = TumorDataset(train_csv, transform=train_transform,max_imgs_per_patient=image_num)
    valid_data = TumorDataset(val_csv, transform=val_transform,max_imgs_per_patient=image_num)

    print("Train Patch:", Counter(train_data.labels))
    print("Val Patch:", Counter(valid_data.labels))

    print(f"Train Patient number: {len(set(train_data.patient_id))}")
    print(f"Val Patient number: {len(set(valid_data.patient_id))}")

    loader_args = {
        'batch_size': 128,
        'pin_memory': True,
        'num_workers': 6,
        'persistent_workers': True
    }

    Trainloader = DataLoader(train_data, shuffle=True, **loader_args)
    Valloader = DataLoader(valid_data, shuffle=False, **loader_args)

    # =========================
    # Model
    # =========================
    model = timm.create_model(
        model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
        dynamic_img_size=True,
        pretrained=True,
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = model.num_features
    num_classes = 4 
    model.head = nn.Sequential(            
        nn.Linear(num_features, num_classes)    
    )

    # Freeze all the layers
    for param in model.parameters():
        param.requires_grad = False
    # Finetune the classifcation layer
    for param in model.head.parameters():
        param.requires_grad = True
    # Finetune the last eight transformer blocks
    for param in model.blocks[-8:].parameters():
        param.requires_grad = True
    # Unfreeze the final LayerNorm
    for param in model.norm.parameters():
        param.requires_grad = True
    #AdamW Optimizer
    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr":5e-5},
        {"params": model.norm.parameters(), "lr":1e-5},
        {"params": model.blocks[-8:].parameters(), "lr":1e-5},
    ])
    model = model.to(device)

    # =========================
    # Loss
    # =========================
    criterion = nn.CrossEntropyLoss()

    # =========================
    # Train config
    # =========================
    epochs = 30
    best_metric = 0.0
    patience = 3
    epochs_since_improvement = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # =========================
    # Training loop
    # =========================
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels, patient_id in Trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        avg_train_loss = train_loss / len(Trainloader)
        train_acc = train_correct / train_total

        avg_val_loss, patch_metrics, patient_metrics = evaluate_model(
            model=model,
            dataloader=Valloader,
            criterion=criterion,
            device=device,
            label_names=label_names,
            num_classes=num_classes,
            plot_cm=False
        )

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f} | Val Loss={avg_val_loss:.4f}")
        print(f"   ↳ [Patch Level]   Acc: {patch_metrics['acc']:.4f} | F1: {patch_metrics['f1']:.4f}")
        print(f"   ↳ [Patient Level] Acc: {patient_metrics['acc']:.4f} | F1: {patient_metrics['f1']:.4f}")

        # f1 score as the main metric for early stopping
        current_metric = patch_metrics['f1']

        if current_metric > best_metric:
            best_metric = current_metric
            epochs_since_improvement = 0
            torch.save(model.state_dict(), "best_model_tumor_4class.pth")
            print(f"Saved best model! (Patient-F1: {best_metric:.4f})")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s).")

        if epochs_since_improvement >= patience:
            print("Early stopping triggered.")
            break

if __name__ == "__main__":
    main()