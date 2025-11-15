import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import cv2

from torchcam.methods import GradCAM

# ----------------- DEVICE -----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 20
classes = [f"Class {i+1}" for i in range(num_classes)]


# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()


# ----------------- GradCAM EXTRACTOR -----------------
# Use last convolution layer of MobileNetV2
cam_extractor = GradCAM(model, target_layer="features.18.0")


# ----------------- TRANSFORM -----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ----------------- PREDICT + CAM -----------------
def predict_and_visualize(image):
    img_t = transform(image).unsqueeze(0).to(device)
    outputs = model(img_t)

    probs = torch.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, 1)
    class_idx = pred.item()

    # --- GradCAM activation map ---
    cam = cam_extractor(class_idx, outputs)[0]  # shape: (H, W) or (1, H, W)
    cam = cam.cpu().numpy()

    # Fix shape problems
    if cam.ndim == 3:
        cam = cam.squeeze()     # (1, H, W) → (H, W)

    # Handle NaN values
    cam = np.nan_to_num(cam, nan=0.0)

    # Normalize safely
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Convert → uint8 0–255
    heatmap = (cam * 255).astype("uint8")   # NOW CV_8UC1

    # Convert base image
    base_img = np.array(image.convert("RGB"))
    H, W, _ = base_img.shape

    # Resize CAM to match image
    heatmap = cv2.resize(heatmap, (W, H))

    # Apply color map (valid)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Overlay
    overlay = cv2.addWeighted(base_img, 0.5, heatmap_color, 0.5, 0)
    overlay_img = Image.fromarray(overlay)

    return classes[class_idx], confidence.item(), overlay_img


# ----------------- UI -----------------
st.title("Object Recognition + GradCAM")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
camera_input = st.camera_input("Camera")

image = None
if uploaded:
    image = Image.open(uploaded)
elif camera_input:
    image = Image.open(camera_input)

if image:
    st.image(image, caption="Input", use_container_width=True)

    if st.button("Predict"):
        label, conf, heatmap = predict_and_visualize(image)

        st.success(f"Class: {label}")
        st.write(f"Confidence: {conf * 100:.2f}%")

        st.image(heatmap, caption="GradCAM", use_container_width=True)
