# scripts/train.py
import os
import json
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, applications, mixed_precision
import matplotlib.pyplot as plt

# =========================
# JSON-safe conversion helper
# =========================
def to_python(obj):
    """
    Convert TensorFlow tensors and NumPy types to plain Python so json.dump/json.dumps won't fail.
    """
    import numpy as np
    # TensorFlow tensor -> NumPy
    if tf.is_tensor(obj):
        obj = obj.numpy()
    # NumPy scalar -> Python scalar
    if isinstance(obj, np.generic):
        return obj.item()
    # NumPy array -> list
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    # Containers: recurse
    if isinstance(obj, (list, tuple)):
        return [to_python(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    return obj

# =========================
# GPU setup
# =========================
# Optional: pick specific GPU(s)
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs detected: {gpus}")
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Logical GPUs: {logical_gpus}")
    except RuntimeError as e:
        print(f"GPU config error: {e}")
else:
    print("No GPU detected. Running on CPU.")

# Mixed precision (recommended on RTX/Ampere+)
mixed_precision.set_global_policy("mixed_float16")

# =========================
# Config
# =========================
# Choose backbone: "mobilenetv3", "mobilenetv2", "efficientnetv2s", "resnet50", "efficientnetb0"
BACKBONE = "mobilenetv3"  # change here if you want a different backbone

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "..", "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Default image size (adjusted below if InceptionV3)
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
SEED = 42

# Training schedule
EPOCHS_HEAD = 50          # Phase 1: train head only
EPOCHS_FT = 100           # Phase 2: fine-tune backbone
LR_HEAD = 1e-3
LR_FT = 2e-5
UNFREEZE_LAST_N = 50     # Unfreeze last N layers during fine-tuning

# =========================
# Backbone factory
# =========================
def build_backbone_and_preprocess(backbone_name: str, input_shape):
    bn = backbone_name.lower()
    if bn == "mobilenetv3":
        base = applications.MobileNetV3Large(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = applications.mobilenet_v3.preprocess_input
    elif bn == "mobilenetv2":
        base = applications.MobileNetV2(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = applications.mobilenet_v2.preprocess_input
    elif bn == "efficientnetv2s":
        base = applications.EfficientNetV2S(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = applications.efficientnet_v2.preprocess_input
    elif bn == "resnet50":
        base = applications.ResNet50(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = applications.resnet.preprocess_input
    elif bn == "efficientnetb0":
        base = applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
        preprocess = applications.efficientnet.preprocess_input
    else:
        raise ValueError(f"Unknown BACKBONE '{backbone_name}'.")
    return base, preprocess

# If using InceptionV3 (not default), set IMAGE_SIZE=(299,299) and switch below
USE_INCEPTION = False  # set True if you want InceptionV3
if USE_INCEPTION:
    IMAGE_SIZE = (299, 299)

# Sanity check image size
assert isinstance(IMAGE_SIZE, tuple) and len(IMAGE_SIZE) == 2 and all(isinstance(v, int) for v in IMAGE_SIZE), f"Bad IMAGE_SIZE: {IMAGE_SIZE}"
input_shape = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
print("Using input_shape:", input_shape)

# =========================
# Data: load, cache, prefetch
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True,
    seed=SEED,
)

# If val directory exists and has files, use it; else split from train
if os.path.isdir(VAL_DIR) and any(os.scandir(VAL_DIR)):
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
        shuffle=False,
        seed=SEED,
    )
else:
    val_fraction = 0.1
    val_batches = max(1, int(len(train_ds) * val_fraction))
    val_ds = train_ds.take(val_batches)
    train_ds = train_ds.skip(val_batches)

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)

# Save class names
with open(os.path.join(OUTPUTS_DIR, "class_names.json"), "w") as f:
    json.dump(to_python(CLASS_NAMES), f, indent=2)

# Cache and prefetch
train_ds = train_ds.cache().prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(tf.data.AUTOTUNE)

# =========================
# Augmentation
# =========================
data_aug = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
], name="data_augmentation")

# =========================
# Model build
# =========================
# Choose backbone
if USE_INCEPTION:
    # InceptionV3 option (requires IMAGE_SIZE=(299,299))
    base_model = applications.InceptionV3(
        include_top=False, weights="imagenet", input_shape=input_shape
    )
    preprocess = applications.inception_v3.preprocess_input
else:
    base_model, preprocess = build_backbone_and_preprocess(BACKBONE, input_shape)

# Phase 1: freeze backbone
base_model.trainable = False

inputs = tf.keras.Input(shape=input_shape)
x = data_aug(inputs)
x = preprocess(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
# With mixed precision, output should be float32 for numerical stability
outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(LR_HEAD),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stopping = callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=5,
    restore_best_weights=True
)
checkpoint_head = callbacks.ModelCheckpoint(
    os.path.join(MODELS_DIR, f"best_model_phase1_{BACKBONE}.h5"),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

print(f"Starting Phase 1 (head-only) training with backbone '{BACKBONE}'...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    callbacks=[early_stopping, checkpoint_head],
    verbose=1
)

# Save Phase 1 history safely
with open(os.path.join(OUTPUTS_DIR, f"history_phase1_{BACKBONE}.json"), "w") as f:
    json.dump(to_python(history.history), f, indent=2)

# =========================
# Phase 2: Fine-tune backbone
# =========================
print(f"Unfreezing last {UNFREEZE_LAST_N} layers of backbone for fine-tuning...")
base_model.trainable = True
if UNFREEZE_LAST_N is not None and UNFREEZE_LAST_N > 0:
    for layer in base_model.layers[:-UNFREEZE_LAST_N]:
        layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(LR_FT),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

checkpoint_ft = callbacks.ModelCheckpoint(
    os.path.join(MODELS_DIR, f"best_model_finetuned_{BACKBONE}.h5"),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

print("Starting Phase 2 (fine-tuning) training...")
history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    callbacks=[early_stopping, checkpoint_ft],
    verbose=1
)

# Save Phase 2 history safely
with open(os.path.join(OUTPUTS_DIR, f"history_phase2_{BACKBONE}.json"), "w") as f:
    json.dump(to_python(history_ft.history), f, indent=2)

# =========================
# Plot metrics
# =========================
plt.figure(figsize=(12, 4))

# Loss
plt.subplot(1, 2, 1)
plt.plot(history.history.get("loss", []), label="Train loss (P1)")
plt.plot(history.history.get("val_loss", []), label="Val loss (P1)")
if "loss" in history_ft.history:
    plt.plot(history_ft.history["loss"], label="Train loss (P2)")
    plt.plot(history_ft.history["val_loss"], label="Val loss (P2)")
plt.legend()
plt.title(f"Loss ({BACKBONE})")

# Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history.get("accuracy", []), label="Train acc (P1)")
plt.plot(history.history.get("val_accuracy", []), label="Val acc (P1)")
if "accuracy" in history_ft.history:
    plt.plot(history_ft.history["accuracy"], label="Train acc (P2)")
    plt.plot(history_ft.history["val_accuracy"], label="Val acc (P2)")
plt.legend()
plt.title(f"Accuracy ({BACKBONE})")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUTS_DIR, f"training_plot_{BACKBONE}.png"))
plt.close()

# =========================
# Save final model
# =========================
final_model_path = os.path.join(MODELS_DIR, f"final_model_{BACKBONE}.h5")
model.save(final_model_path)
print(f"Saved final model to: {final_model_path}")
print("Training complete.")
