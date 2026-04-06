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
    # 保持不变
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

# --- 2. 优化 Dataset 类 ---
class TumorDataset(Dataset):
    def __init__(self, image_root, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform
        self.label_mapping = {"gbm": 0, "astro": 1, "oligo": 2}
        for tumor_type, label in self.label_mapping.items():
            class_path = os.path.join(image_root, tumor_type)
            if not os.path.exists(class_path): continue
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
    
# --- 1. 数据增强 (Transforms) ----
train_transform = v2.Compose([
    v2.ToImage(),
    v2.RandomResizedCrop(224, scale=(0.8, 1.0)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.RandomVerticalFlip(p=0.5),
    v2.RandomRotation(degrees=180),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
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

# ==========================================
# 新增：Multi-Head 模型结构定义
# ==========================================
class MultiHeadViT(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.encoder = base_model
        self.num_features = self.encoder.num_features
        self.encoder.head = nn.Identity() # 移除原有的 3 分类头

        # Head 1: 二分类 -> 0: GBM, 1: Non-GBM (Astro or Oligo)
        self.head1 = nn.Linear(self.num_features, 2)
        
        # Head 2: 二分类 -> 0: Astro, 1: Oligo
        self.head2 = nn.Linear(self.num_features, 2)

    def forward(self, x):
        feat = self.encoder(x)
        out1 = self.head1(feat)
        out2 = self.head2(feat)
        return out1, out2

# ==========================================
# 修改：双头评估逻辑
# ==========================================
def evaluate_test_set(model, test_loader, device, label_mapping):
    model.eval()
    all_preds = []
    all_labels = []
    
    inv_label_mapping = {v: k for k, v in label_mapping.items()}
    class_names = [inv_label_mapping[i] for i in range(len(label_mapping))]

    print("开始测试集评估...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out1, out2 = model(images)
            
            _, pred_h1 = torch.max(out1, 1)
            _, pred_h2 = torch.max(out2, 1)
            
            # --- 核心推理逻辑 ---
            # 初始化一个预测数组
            final_preds = torch.zeros_like(pred_h1)
            
            # 规则 1: 如果 Head 1 说是 GBM (0)，那就是 GBM (0)
            final_preds[pred_h1 == 0] = 0
            
            # 规则 2: 如果 Head 1 说是 Non-GBM (1)，那就看 Head 2 的结果
            # Head 2 输出 0 代表 Astro(原标签1), 输出 1 代表 Oligo(原标签2)
            mask_non_gbm = (pred_h1 == 1)
            final_preds[mask_non_gbm] = pred_h2[mask_non_gbm] + 1
            
            all_preds.extend(final_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    correct = (np.array(all_preds) == np.array(all_labels)).sum()
    total = len(all_labels)
    test_acc = correct / total
    print(f"\n===== 测试结果 =====")
    print(f"总准确率 (Overall Accuracy): {test_acc:.4f}")

    print("\n详细分类报告:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
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
        np.arange(len(labels)), test_size=0.2, random_state=42, stratify=labels
    )

    train_data = ApplyTransformSubset(Subset(full_train_dataset, train_indices), transform=train_transform)
    valid_data = ApplyTransformSubset(Subset(full_train_dataset, val_indices), transform=val_transform)
    test_data = TumorDataset(test_root, transform=val_transform)

    train_labels_subset = [labels[i] for i in train_indices]
    class_counts = Counter(train_labels_subset)
    print(f"训练集类别分布: {class_counts}")
    
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[lbl] for lbl in train_labels_subset]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    loader_args = {'batch_size': 128, 'pin_memory': True, 'num_workers': 8, 'persistent_workers': True}

    Trainloader = DataLoader(train_data, sampler=sampler, **loader_args)
    Valloader = DataLoader(valid_data, shuffle=False, **loader_args)
    Testloader = DataLoader(test_data, shuffle=False, **loader_args)

    # --- 4. 初始化 Multi-Head 模型 ---
    base_model = timm.create_model(
        model_name="hf-hub:1aurent/vit_base_patch16_224.kaiko_ai_towards_large_pathology_fms",
        pretrained=True,
    )
    
    model = MultiHeadViT(base_model).to(device)

    # 冻结与解冻逻辑
    for param in model.encoder.parameters(): param.requires_grad = False
    for param in model.head1.parameters(): param.requires_grad = True
    for param in model.head2.parameters(): param.requires_grad = True
    for param in model.encoder.blocks[-4:].parameters(): param.requires_grad = True
    for param in model.encoder.norm.parameters(): param.requires_grad = True

    # 优化器包含两个 Head
    optimizer = torch.optim.AdamW([
        {"params": model.head1.parameters(), "lr": 1e-4},
        {"params": model.head2.parameters(), "lr": 1e-4},
        {"params": model.encoder.blocks[-4:].parameters(), "lr": 5e-6},
        {"params": model.encoder.norm.parameters(), "lr": 5e-6},
    ], weight_decay=0.05)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    criteria = nn.CrossEntropyLoss()

    epochs = 10
    best_acc = 0.0
    print("start training Multi-Head Model...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for images, labels in Trainloader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # --- 构建双头 Target ---
            # labels 原值为: 0 (GBM), 1 (Astro), 2 (Oligo)
            
            # Head 1: 0 归为 0, 1和2 归为 1
            labels_h1 = (labels > 0).long()
            
            # Head 2: 提取非 GBM 的样本进行计算
            valid_h2_mask = (labels > 0)
            # 把 1(Astro) 映射为 0, 把 2(Oligo) 映射为 1
            labels_h2 = labels[valid_h2_mask] - 1 

            optimizer.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out1, out2 = model(images)
                
                # 计算 Head 1 的 Loss
                loss1 = criteria(out1, labels_h1)
                
                # 计算 Head 2 的 Loss (只计算不是 GBM 的样本)
                if valid_h2_mask.sum() > 0:
                    loss2 = criteria(out2[valid_h2_mask], labels_h2)
                    loss = loss1 + loss2 # 将两部分 Loss 1:1 相加
                else:
                    loss = loss1

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images_v, labels_v in Valloader:
                images_v, labels_v = images_v.to(device), labels_v.to(device)
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    out1_v, out2_v = model(images_v)
                    
                _, pred_h1_v = out1_v.max(1)
                _, pred_h2_v = out2_v.max(1)
                
                # 构建最终预测，用于验证集打分
                final_preds_v = torch.zeros_like(pred_h1_v)
                final_preds_v[pred_h1_v == 0] = 0
                mask_non_gbm_v = (pred_h1_v == 1)
                final_preds_v[mask_non_gbm_v] = pred_h2_v[mask_non_gbm_v] + 1
                
                val_correct += (final_preds_v == labels_v).sum().item()
                val_total += labels_v.size(0)

        val_acc = val_correct / val_total
        print(f"Epoch {epoch+1}: Train Loss (Total): {train_loss/len(Trainloader):.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_multihead.pth")
            print("Model Saved!")

    print("加载最佳 Multi-Head 模型权重进行测试...")
    model.load_state_dict(torch.load("best_model_multihead.pth"))
    
    label_mapping = {"gbm": 0, "astro": 1, "oligo": 2}
    evaluate_test_set(model, Testloader, device, label_mapping)

if __name__ == "__main__":
    main()