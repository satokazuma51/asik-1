import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

DATASET_DIR = "C:/Users/Nurik/Desktop/dataset"  # <-- your dataset
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
SEED = 42

datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=25,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.8, 1.2),
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
    seed=SEED
)

NUM_CLASSES = len(train_gen.class_indices)
class_names = list(train_gen.class_indices.keys())
print("Detected classes:", class_names)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()

checkpoint_cb = callbacks.ModelCheckpoint("best_model_10classes.h5", save_best_only=True, monitor="val_accuracy")
earlystop_cb = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

EPOCHS_HEAD = 10
history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=1
)

base_model.trainable = True
fine_tune_at = int(len(base_model.layers) * 0.7)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss="categorical_crossentropy", metrics=["accuracy"])

EPOCHS_FINE = 10
history2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD + EPOCHS_FINE,
    initial_epoch=history1.epoch[-1] + 1,
    callbacks=[checkpoint_cb, earlystop_cb],
    verbose=1
)

val_steps = int(np.ceil(val_gen.samples / val_gen.batch_size))
y_prob = model.predict(val_gen, steps=val_steps)
y_pred = np.argmax(y_prob, axis=1)
y_true = val_gen.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix - MobileNetV2 (10 Classes)")
plt.show()

batch_imgs, batch_labels = next(iter(val_gen))
num_to_show = min(10, len(batch_imgs))
plt.figure(figsize=(15, 6))
for i in range(num_to_show):
    image = batch_imgs[i]
    true_idx = np.argmax(batch_labels[i])
    p = model.predict(image[np.newaxis, ...], verbose=0)
    pred_idx = np.argmax(p)
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.title(f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]}")
    plt.axis("off")
plt.suptitle("Sample Predictions - MobileNetV2 (10 Classes)")
plt.show()

print("✅ Model training complete. Saved as best_model_10classes.h5")
