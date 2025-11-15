import os
import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(args.data_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    num_classes = len(val_dataset.classes)
    print(f"Number of classes: {num_classes}")

    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    print(f"Validation accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Household Object Recognition Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model file")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset folder with class subfolders")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()
    evaluate(args)
