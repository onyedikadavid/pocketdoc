# api_server.py
import os
# Force TensorFlow to run on CPU to avoid CUDA initialization errors on CPU platforms like Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import base64
import io
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from class_mappings import CLASS_NAMES, CLEAN_LABELS, determine_triage_level
from medical_search import fetch_dynamic_disease_info  # Dynamic insights import
from chat import router as chat_router  # Imported pre-consultation chat router
from intake import router as intake_router  # Doctor handoff / intake transfer router

app = FastAPI(title="PocketDoc AI Microservice", version="1.0")

# Enable CORS for Next.js proxy calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the pre-consultation chatbot routes
app.include_router(chat_router, prefix="/chat", tags=["Pre-Consultation Chat"])

# Mount the doctor intake-transfer route
app.include_router(intake_router, prefix="/intake", tags=["Doctor Handoff"])

# Load Trained Model
MODEL_PATH = Path("models/skin_disease_model.keras")
print(f"Loading Keras Model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


def find_target_conv_layer(model: tf.keras.Model) -> str:
    """Finds the last 2D/4D convolutional layer safely without crashing on non-spatial layers."""
    # First check layer instance types (standard for Keras/TensorFlow models)
    for layer in reversed(model.layers):
        if isinstance(layer, (tf.keras.layers.Conv2D, tf.keras.layers.DepthwiseConv2D)):
            return layer.name

    # Fallback: check output tensor shape rank safely
    for layer in reversed(model.layers):
        try:
            if hasattr(layer, "output") and len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue

    raise ValueError("No 2D/4D convolutional layer found in model architecture for Grad-CAM.")


TARGET_CONV_LAYER = find_target_conv_layer(model)


def generate_gradcam_heatmap(img_array: np.ndarray, target_class_idx: int) -> str:
    """
    Generates a Grad-CAM heatmap for the target class prediction and returns
    a Base64 encoded JPEG string overlaid onto the original image.
    """
    try:
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(TARGET_CONV_LAYER).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, target_class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
        heatmap = heatmap.numpy()

        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        original_img = np.uint8(img_array[0] * 255)
        original_bgr = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)

        overlay = cv2.addWeighted(original_bgr, 0.6, colored_heatmap, 0.4, 0)
        overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(overlay_rgb)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG")
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return f"data:image/jpeg;base64,{base64_str}"
    except Exception as e:
        print(f"Grad-CAM Heatmap generation failed: {str(e)}")
        return ""


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes to a preprocessed tensor for EfficientNetB0."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img, dtype=np.float32)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        raise ValueError(f"Failed to process image payload: {str(e)}")


@app.get("/")
def health_check():
    return {"status": "online", "model": "EfficientNetB0 (23 classes)"}


@app.post("/predict")
async def predict_skin_disease(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")

    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image_bytes(image_bytes)

        # Run Prediction Inference
        predictions = model.predict(input_tensor, verbose=0)[0]

        # Get Top-3 Predictions, highest confidence first
        top_3_indices = np.argsort(predictions)[-3:][::-1]

        top_3_predictions = []
        for idx in top_3_indices:
            raw_class_name = CLASS_NAMES[idx]
            clean_label = CLEAN_LABELS.get(raw_class_name, raw_class_name)
            confidence_score = float(predictions[idx])

            top_3_predictions.append({
                "class_name": clean_label,
                "confidence": round(confidence_score, 4)
            })

        top_class_idx = int(top_3_indices[0])
        top_raw_class = CLASS_NAMES[top_class_idx]
        top_confidence = float(predictions[top_class_idx])
        top_clean_label = top_3_predictions[0]["class_name"]

        triage_level = determine_triage_level(top_raw_class, top_confidence)

        # Generate Grad-CAM Visual Heatmap
        heatmap_image = generate_gradcam_heatmap(input_tensor, top_class_idx)

        # Dynamically fetch medical insight details for ANY predicted disease
        medical_insights = fetch_dynamic_disease_info(top_clean_label)

        return {
            "success": True,
            "top_prediction": top_3_predictions[0],
            "top_3_predictions": top_3_predictions,
            "triage_level": triage_level,
            "explanation": {
                "heatmap_image": heatmap_image,
                "description": f"Visual features highlighted in red/yellow contributed most to the '{top_clean_label}' prediction.",
                "insights": medical_insights
            }
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)