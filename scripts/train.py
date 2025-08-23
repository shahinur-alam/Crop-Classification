# scripts/train.py
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications
import matplotlib.pyplot as plt
import pandas as pd
import os

# Config
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
NUM_CLASSES = 5
CLASS_NAMES = ["Sunflower", "Paddy", "Nuts", "Lentil", "Capsicum"]
MODEL_NAME = "efficientnetb0"
FREEZE_BACKBONE = False
OUTPUT_DIR = "../models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data
train_ds = tf.keras.utils.image_dataset_from_directory(
    '../data/train',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=42,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    '../data/val',
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False,
    seed=42,
)

# Optional: Data augmentation
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.1)
])

# Model
#base_model = applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=(*IMAGE_SIZE, 3))

base_model = applications.EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)   # <-- This is correct for TF 2.x
)


if FREEZE_BACKBONE:
    base_model.trainable = False

inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = data_aug(inputs)
x = applications.efficientnet.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = models.Model(inputs, outputs)

# Compile
model.compile(
    optimizer=optimizers.Adam(1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
checkpoint = callbacks.ModelCheckpoint(
    os.path.join(OUTPUT_DIR, 'best_model.h5'),
    save_best_only=True,
    monitor='val_accuracy'
)

# Train
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint]
)

# Plot
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train loss')
plt.plot(history.history['val_loss'], label='Val loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.legend()
plt.savefig('../outputs/training_plot.png')
plt.close()

# Save final model
model.save(os.path.join(OUTPUT_DIR, 'final_model.h5'))
