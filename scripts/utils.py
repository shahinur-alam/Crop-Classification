# scripts/utils.py
from tensorflow.keras.preprocessing import image
import numpy as np

def load_and_preprocess_image(path, target_size=(224, 224)):
    img = image.load_img(path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return img_array, img.filename
