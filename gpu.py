# scripts/train.py
import tensorflow as tf
import os

import tensorflow as tf
print(tf.__version__)


# List and configure GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Prevent TF from allocating all VRAM at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

        # Optional: set a per-process VRAM cap (in MB) on first GPU instead of growth
        # tf.config.set_logical_device_configuration(
        #     gpus[0],
        #     [tf.config.LogicalDeviceConfiguration(memory_limit=8192)]  # ~8GB
        # )

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"GPUs detected: {gpus}")
        print(f"Logical GPUs: {logical_gpus}")
    except RuntimeError as e:
        # Must be set before GPUs are initialized
        print(f"GPU config error: {e}")
else:
    print("No GPU detected. Running on CPU.")
