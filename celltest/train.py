import os
import torch
import torch.nn as nn
import timm
import numpy as np
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F


class SoftFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.1, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.alpha = alpha

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=-1)

        if targets.dim() == 1:
            targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1)).float()
        else:
            targets_one_hot = targets.float()

        if self.label_smoothing > 0:
            targets_one_hot = targets_one_hot * (1 - self.label_smoothing) + self.label_smoothing / inputs.size(-1)

        pt = torch.sum(targets_one_hot * probs, dim=-1)
        ce_loss = -torch.sum(targets_one_hot * F.log_softmax(inputs, dim=-1), dim=-1)

        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha_tensor = torch.tensor(self.alpha, device=inputs.device)
            alpha_weight = torch.sum(targets_one_hot * alpha_tensor, dim=-1)
            loss = alpha_weight * loss

        return loss.mean()


class TumorDataset(Dataset):
    def __init__(self, image_root, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform

        # 二分类:
        # gbm -> 0
        # astro + oligo -> 1 (idh mutant)
        self.label_mapping = {
            "gbm": 0,
            "astro": 1,
            "oligo": 1,
        }

        for tumor_type, label in self.label_mapping.items():
            class_path = os.path.join(image_root, tumor_type)
            if not os.path.exists(class_path):
                continue

            for root, dirs, files in os.walk(class_path):
                for img_name in files:
                    if img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        self.images.append(os.path.join(root, img_name))
                        self.labels.append(label)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.images)


class ApplyTransformSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(size=224, scale=(0.7, 1.3), ratio=(0.9, 1.1)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(180),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    v2.RandomGrayscale(p=0.05),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

val_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize(224),
    v2.CenterCrop(224),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

# 改成二分类
mixup_cutmix = v2.RandomChoice([
    v2.CutMix(alpha=1.0, num_classes=2),
    v2.MixUp(alpha=0.2, num_classes=2)
])


def evaluate_test_set(model, test_loader, device, label_mapping):
    model.eval()
    all_preds = []
    all_labels = []

    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [inv_label_mapping[i] for i in range(len(inv_label_mapping))]

    print("开始测试集评估...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct = (np.array(all_preds) == np.array(all_labels)).sum()
    total = len(all_labels)
    test_acc = correct / total

    print(f"\n===== 测试结果 =====")
    print(f"总准确率 (Overall Accuracy): {test_acc:.4f}")

    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def main():
    torch.set_float32_matmul_precision('high')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_root = r"D:\Downloads\train_glioma\train"
    test_root = r"D:\Downloads\test\test"

    full_train_dataset = TumorDataset(train_root)
    labels = full_train_dataset.labels

    train_indices, val_indices = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    train_data = ApplyTransformSubset(Subset(full_train_dataset, train_indices), transform=train_transform)
    valid_data = ApplyTransformSubset(Subset(full_train_dataset, val_indices), transform=val_transform)
    test_data = TumorDataset(test_root, transform=val_transform)

    train_labels_subset = [labels[i] for i in train_indices]
    class_counts = Counter(train_labels_subset)
    print(f"训练集类别分布: {class_counts}")

    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels_subset]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    loader_args = {
        'batch_size': 128,
        'pin_memory': True,
        'num_workers': 8,
        'persistent_workers': True
    }

    Trainloader = DataLoader(train_data, sampler=sampler, **loader_args)
    Valloader = DataLoader(valid_data, shuffle=False, **loader_args)
    Testloader = DataLoader(test_data, shuffle=False, **loader_args)

    print(f"训练集: {len(train_data)} | 验证集: {len(valid_data)} | 测试集: {len(test_data)}")
    print(f"训练集原始分布: {class_counts}")

    model = timm.create_model(
        model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
        pretrained=True,
    )

    num_features = model.num_features
    model.head = nn.Linear(num_features, 2)   # 改成2类

    for param in model.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.blocks[-4:].parameters():
        param.requires_grad = True
    for param in model.norm.parameters():
        param.requires_grad = True

    model = model.to(device)

    optimizer = torch.optim.AdamW([
        {"params": model.head.parameters(), "lr": 1e-4},
        {"params": model.blocks[-4:].parameters(), "lr": 5e-6},
        {"params": model.norm.parameters(), "lr": 5e-6},
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # 二分类 alpha 也改成 2 个值
    criteria = SoftFocalLoss(
        gamma=2.0,
        label_smoothing=0.0,
        alpha=[1.0, 1.2]   # 可按类别不平衡再调
    )

    epochs = 10
    best_acc = 0.0

    print("start training...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for images, labels in Trainloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 如果你想启用 MixUp/CutMix，再打开这两行
            # images, labels = mixup_cutmix(images, labels)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(images)
                loss = criteria(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)

            # 如果开启 MixUp/CutMix，这里准确率不能这样直接算
            if labels.dim() == 1:
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images_v, labels_v in Valloader:
                images_v = images_v.to(device)
                labels_v = labels_v.to(device)

                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs_v = model(images_v)

                _, predicted_v = outputs_v.max(1)
                val_correct += (predicted_v == labels_v).sum().item()
                val_total += labels_v.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(Trainloader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_binary.pth")
            print("Model Saved!")

    print("加载最佳模型权重进行测试...")
    model.load_state_dict(torch.load("best_model_binary.pth"))

    label_mapping = {
        "gbm": 0,
        "idh_mutant": 1
    }
    evaluate_test_set(model, Testloader, device, label_mapping)


if __name__ == "__main__":
    main()