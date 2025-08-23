import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras import applications
from PIL import Image
import pandas as pd
from glob import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
DEFAULT_IMAGE_DIR = os.path.join(DATA_DIR, "test")  # <- change if you want

# Optional: load saved class names if you wrote them during training
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "..", "outputs", "class_names.json")

def load_class_names(train_dir_fallback):
    # Try reading saved class names; fallback to train folder discovery
    if os.path.isfile(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            return json.load(f)
    ds = tf.keras.utils.image_dataset_from_directory(
        train_dir_fallback, image_size=(224, 224), batch_size=1
    )
    return ds.class_names

def predict_dir(image_dir, model_path, class_names):
    rows = []
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files.extend(glob(os.path.join(image_dir, ext)))
    files = sorted(files)

    if not files:
        print(f"No images found in: {image_dir}")
        return pd.DataFrame([])

    model = tf.keras.models.load_model(model_path)

    for fp in files:
        img = Image.open(fp).convert("RGB").resize((224, 224))
        x = np.array(img)[None, ...]
        x = applications.efficientnet.preprocess_input(x)
        pred = model.predict(x, verbose=0)[0]
        idx = int(np.argmax(pred))
        rows.append({
            "image": fp,
            "prediction": class_names[idx],
            "confidence": float(pred[idx])
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,  # <- default path
        help="Folder containing images to classify (defaults to ../data/test_unlabeled)"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=os.path.join(MODELS_DIR, "best_model.h5"),
        help="Path to the trained model"
    )
    args = parser.parse_args()

    class_names = load_class_names(os.path.join(DATA_DIR, "train"))
    df = predict_dir(args.image_dir, args.model_path, class_names)

    os.makedirs(os.path.join(BASE_DIR, "..", "outputs"), exist_ok=True)
    out_csv = os.path.join(BASE_DIR, "..", "outputs", "unlabeled_predictions.csv")
    if not df.empty:
        df.to_csv(out_csv, index=False)
        print(f"Saved predictions to: {out_csv}")
