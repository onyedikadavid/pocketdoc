import tensorflow as tf

# 1. Point directly to your model file in the models directory
model_path = "models/skin_disease_model.keras"

# 2. Load the trained Keras model
print(f"Loading model from {model_path}...")
model = tf.keras.models.load_model(model_path)

# 3. Convert to TFLite format
print("Converting model to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# 4. Save the lightweight TFLite model inside the models folder
output_path = "models/skin_model.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"Successfully converted and saved to {output_path}")