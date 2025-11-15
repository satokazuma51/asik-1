import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset
import random

# ArtAug: image enhancement wrapper
def artaug(image: Image.Image) -> Image.Image:
    enhanced = image.copy()
    enhanced = ImageEnhance.Brightness(enhanced).enhance(1.3)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.2)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.4)
    return enhanced

# Step-Video-T2V integration stub (pseudo-code placeholder)
# Real Step-Video-T2V model usage requires setup and pretrained weights from Neurohive repo
def generate_synthetic_video(image: Image.Image):
    """
    Given an input image, generate a synthetic video (frames with motion/blur)
    using Step-Video-T2V.
    Here, we'll just simulate by applying motion blur in multiple directions.
    """
    frames = []
    img_np = np.array(image.convert('RGB'))

    def motion_blur(img, degree=15, angle=45):
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        kernel = np.diag(np.ones(degree))
        kernel = cv2.warpAffine(kernel, M, (degree, degree))
        kernel = kernel / degree
        blurred = cv2.filter2D(img, -1, kernel)
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        return np.array(blurred, dtype=np.uint8)

    angles = np.linspace(0, 360, num=8)
    for a in angles:
        frame = motion_blur(img_np, degree=15, angle=a)
        frames.append(Image.fromarray(frame))

    return frames  # list of PIL Images

# Sample frames (already done by generate_synthetic_video)
# You can save them or use directly for training augmentation

# Custom Dataset that applies ArtAug and includes synthetic frames
class AugmentedDataset(Dataset):
    def __init__(self, root_dir, transform=None, apply_artaug=True, synthetic_frames_per_img=3):
        self.dataset = ImageFolder(root_dir)
        self.transform = transform
        self.apply_artaug = apply_artaug
        self.synthetic_frames_per_img = synthetic_frames_per_img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        imgs = [img]

        # Add synthetic frames from Step-Video-T2V (motion blur simulation)
        synthetic_frames = generate_synthetic_video(img)[:self.synthetic_frames_per_img]
        imgs.extend(synthetic_frames)

        augmented_imgs = []
        for im in imgs:
            if self.apply_artaug:
                im = artaug(im)
            if self.transform:
                im = self.transform(im)
            augmented_imgs.append(im)

        # For simplicity, return list of images with the same label
        return augmented_imgs, label

# Example transform for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Use DataLoader with custom collate function to handle list of augmented images per sample
def collate_augmented(batch):
    imgs = []
    labels = []
    for augmented_imgs, label in batch:
        imgs.extend(augmented_imgs)
        labels.extend([label] * len(augmented_imgs))
    return torch.stack(imgs), torch.tensor(labels)

# Usage:
# dataset = AugmentedDataset('path_to_your_dataset', transform=train_transform)
# dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_augmented)
