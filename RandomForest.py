import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.utils import img_to_array, load_img

# dataset/
# ├── Cups/
# ├── Keys/
# ├── Sofa/
# ├── Table/
# ├── Washer/
# ├── Fridge/
# ├── Chairs/
# ├── Vase/
# ├── Bed/
# └── Oven/

DATASET_DIR = "C:/Users/Nurik/Desktop/dataset"  
IMG_SIZE = (64, 64)

def load_dataset(dataset_path, img_size):
    X, y = [], []
    for label in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, label)
        if not os.path.isdir(class_path):
            continue
        for file in os.listdir(class_path):
            img_path = os.path.join(class_path, file)
            try:
                img = load_img(img_path, target_size=img_size)
                img_array = img_to_array(img) / 255.0
                X.append(img_array.flatten())
                y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return np.array(X), np.array(y)

X, y = load_dataset(DATASET_DIR, IMG_SIZE)
print("Dataset shape:", X.shape, y.shape)
print("Classes found:", np.unique(y))

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - Random Forest (10 Classes)")
plt.show()

import random
sample_idx = random.sample(range(len(X)), min(10, len(X)))
plt.figure(figsize=(12, 5))
for i, idx in enumerate(sample_idx):
    img = X[idx].reshape(*IMG_SIZE, 3)
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"{y[idx]}")
    plt.axis("off")
plt.suptitle("Sample Images from Dataset")
plt.show()
