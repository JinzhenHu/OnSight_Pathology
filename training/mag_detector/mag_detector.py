import os
import timm
import torch
import random
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset,WeightedRandomSampler,DataLoader,Subset
from torch.utils.data import Dataset, DataLoader, Subset
from utlis import TumorDataset, TransformedSubset, evaluate_patch_level
from utlis import train_transform, val_transform
from collections import Counter

def main():
    torch.backends.cudnn.benchmark = True
    root_path = r"D:\UofT\2025fall\OnSight\Revisions\Mag_Detector\OnSight_Pathology"
    label_mapping = {"5x": 0, "10x": 1, "20x": 2, "40x": 3}
    full_dataset = TumorDataset(root_path, label_mapping=label_mapping, transform=None,max_cases = None)

    patients = list(set(full_dataset.slide_ids))   # 3666 patients
    random.seed(42)
    random.shuffle(patients)

    n_total = len(patients)
    n_train = int(0.7 * n_total)
    n_val = int(0.3 * n_total)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train+n_val])

    train_idx = [i for i, pid in enumerate(full_dataset.slide_ids) if pid in train_patients]
    val_idx = [i for i, pid in enumerate(full_dataset.slide_ids) if pid in val_patients]



    train_data = TransformedSubset(Subset(full_dataset, train_idx), transform=train_transform)
    valid_data = TransformedSubset(Subset(full_dataset, val_idx), transform=val_transform)

    print("Train:", Counter([full_dataset.labels[i] for i in train_idx]))
    print("Val:", Counter([full_dataset.labels[i] for i in val_idx]))


    print(f"Train Slide Number: {len(set([full_dataset.slide_ids[i] for i in train_idx]))}")
    print(f"Validation Slide Number: {len(set([full_dataset.slide_ids[i] for i in val_idx]))}")
    loader_args = {
        'batch_size': 256,
        'pin_memory': True,
        'num_workers': 8,
        'persistent_workers': True
    }
    Trainloader = DataLoader(train_data, shuffle=True, **loader_args)
    Valloader = DataLoader(valid_data, shuffle=False, **loader_args)

    device = torch.device("cuda")

    model = timm.create_model(
    model_name="hf-hub:1aurent/vit_small_patch16_224.kaiko_ai_towards_large_pathology_fms",
    # dynamic_img_size=True,
    pretrained=True,
    )

    num_features = model.num_features
    num_classes = len(label_mapping)

    model.head = nn.Sequential(
        nn.Linear(num_features, 1024),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes)
    )

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True


    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": 1e-3},
    ], weight_decay=1e-4) 
    model = model.to(device)

    criteria = nn.CrossEntropyLoss()

    epochs = 30
    best_acc = 0.0
    patience = 5
    epochs_since_improvement = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels, slide_ids in Trainloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criteria(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += (predicted == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()

        avg_train_loss = train_loss / len(Trainloader)
        train_acc = train_correct / train_total

        # Validation Phase 
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images_val, labels_val, slide_ids in Valloader:
                images_val, labels_val = images_val.to(device, non_blocking=True), labels_val.to(device, non_blocking=True)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs_val = model(images_val)
                    loss_val = criteria(outputs_val, labels_val)

                val_loss += loss_val.item()
                _, predicted_val = outputs_val.max(1)
                val_correct += (predicted_val == labels_val).sum().item()
                val_total += labels_val.size(0)

        avg_val_loss = val_loss / len(Valloader)
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.4f} | "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.4f}")
        all_labels, all_preds = evaluate_patch_level(model, Valloader, device, label_names=list(label_mapping.keys()), plot_cm=False)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_since_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best model! (Acc: {best_acc:.4f})")
        else:
            epochs_since_improvement += 1
            print(f"No improvement for {epochs_since_improvement} epoch(s).")

        if epochs_since_improvement >= patience:
            print("Early stopping triggered.")
            break


if __name__ == "__main__":
    main()