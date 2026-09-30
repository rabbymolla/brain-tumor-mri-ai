"""
Retrain script: Combine original data + user corrections, then retrain model.
Usage: python retrain.py
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18
import timm

# ─── Config ───
DATA_DIR = "data/archive/Training"          # Original training data
CORRECTIONS_DIR = "corrections"              # User feedback images
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = "results/brain_tumor_resnet18_retrained.pth"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─── Data Transforms ───
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ─── Load Original + Correction Data ───
print("Loading datasets...")

# Original data
original_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

# Corrections data (if exists)
# Check if corrections have actual images
def has_images(folder_path):
    if not os.path.exists(folder_path):
        return False
    for root, dirs, files in os.walk(folder_path):
        if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in files):
            return True
    return False

# Corrections data (only if has images)
if has_images(CORRECTIONS_DIR):
    corrections_dataset = datasets.ImageFolder(CORRECTIONS_DIR, transform=transform)
    # Combine both datasets
    combined_dataset = torch.utils.data.ConcatDataset([original_dataset, corrections_dataset])
    print(f"Original: {len(original_dataset)} images | Corrections: {len(corrections_dataset)} images")
else:
    combined_dataset = original_dataset
    print(f"Original: {len(original_dataset)} images | No corrections yet")

# Split train/val
train_size = int(0.9 * len(combined_dataset))
val_size = len(combined_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(combined_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

# ─── Model ───
print("Loading model...")
model = resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 4)
model = model.to(DEVICE)

# Load existing weights if available
existing_model = "results/brain_tumor_resnet18.pth"
if os.path.exists(existing_model):
    model.load_state_dict(torch.load(existing_model, map_location=DEVICE))
    print("Loaded existing model weights")

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ─── Training ───
print("\nStarting training...")
best_acc = 0.0

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")
    
    # Save best model
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"  -> Saved best model (acc: {acc:.2f}%)")

print(f"\n✅ Retraining complete! Best accuracy: {best_acc:.2f}%")
print(f"Model saved to: {MODEL_SAVE_PATH}")