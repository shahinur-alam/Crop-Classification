# scripts/evaluate.py
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Load test set
test_ds = tf.keras.utils.image_dataset_from_directory(
    '../data/test',
    image_size=(224, 224),
    batch_size=32,
    label_mode='categorical',
    shuffle=False,
)
class_names = test_ds.class_names

# Load best model
model = tf.keras.models.load_model('../models/final_model_mobilenetv3.h5')

# Evaluate
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")

# Predictions
y_true = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
y_true = np.argmax(y_true, axis=1)
y_pred = np.argmax(model.predict(test_ds), axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# Save predictions
file_paths = [str(p) for p in test_ds.file_paths]
df = pd.DataFrame({
    'path': file_paths,
    'pred': [class_names[i] for i in y_pred],
    'true': [class_names[i] for i in y_true],
})
df.to_csv('../outputs/test_predictions.csv', index=False)
