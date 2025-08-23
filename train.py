import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import os

# Define paths to dataset
train_dir = 'dataset/train'  # Path to training data (organized per crop folder)
val_dir = 'dataset/val'      # Path to validation data (organized per crop folder)
test_dir = 'dataset/test'    # Path to test data (organized per crop folder)

# Parameters
IMG_SIZE = 224  # Image size for MobileNetV2
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
CLASS_NAMES = sorted(os.listdir(train_dir))  # Automatically get class names

# Data preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize images
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
val_datagen = ImageDataGenerator(rescale=1./255)  # Validation data doesn't need augmentation

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Load pretrained MobileNetV2 model, exclude top layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Add custom classifier head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # Add a fully connected layer
predictions = Dense(len(CLASS_NAMES), activation='softmax')(x)  # Output layer for classification
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model (optional, for fine-tuning)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# Fine-tune the model (unfreeze some base model layers)
for layer in base_model.layers[len(base_model.layers) // 2:]:
    layer.trainable = True

# Recompile and continue training
model.compile(optimizer=Adam(LEARNING_RATE / 10), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=val_generator.samples // BATCH_SIZE
)

# Test the model
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=1,
    class_mode='categorical',
    shuffle=False
)

# Evaluate on the test dataset
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Make predictions
import numpy as np
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
for i, img_path in enumerate(test_generator.filenames):
    print(f"Image: {img_path} -> Predicted: {CLASS_NAMES[predicted_classes[i]]}")