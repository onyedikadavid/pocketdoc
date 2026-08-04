# src/predict.py
import numpy as np
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import config

def predict_image(image_path, model_path, class_names):
    model = tf.keras.models.load_model(model_path)
    
    img = image.load_img(image_path, target_size=config.IMAGE_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top_idx = np.argmax(preds)
    confidence = preds[top_idx]

    return class_names[top_idx], float(confidence)