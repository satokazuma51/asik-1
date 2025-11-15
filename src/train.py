import os
import argparse
from PIL import ImageEnhance, Image
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.ops import box_iou
import numpy as np

# Simple ArtAug function to enhance brightness, contrast, sharpness
def artaug(image):
    enhanced = image.copy()
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.3)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.4)
    return enhanced

# Custom Dataset wrapper to apply ArtAug optionally
class AugmentedDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, apply_aug=False):
        super().__init__(root, transform=transform)
        self.apply_aug = apply_aug

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.apply_aug:
            image = artaug(image)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

def calculate_map(outputs, targets, num_classes):
    """
    Dummy mAP@0.5 calculator for classification.
    Replace with proper detection mAP if doing detection task.
    """
    preds = torch.argmax(outputs, dim=1)
    correct = preds.eq(targets).sum().item()
    accuracy = correct / targets.size(0)
    return accuracy  # For classification, accuracy used as proxy

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Datasets
    train_dataset = AugmentedDataset(args.data_dir, transform=train_transform, apply_aug=True)
    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")

    # Model
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # Validation
        model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                all_outputs.append(outputs)
                all_labels.append(labels)

        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)
        acc = calculate_map(all_outputs, all_labels, num_classes)

        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {epoch_loss:.4f} - Validation Accuracy: {acc:.4f}")

        # Save best model
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), args.save_path)
            print(f"Saved best model with accuracy: {best_acc:.4f}")

    print(f"Training completed. Best accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Household Object Recognition Model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder with class subfolders")
    parser.add_argument("--save_path", type=str, default="models/best_model.pth", help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    args = parser.parse_args()
    train(args)
