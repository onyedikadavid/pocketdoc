import os
# Force TensorFlow to run on CPU to avoid CUDA initialization errors on CPU platforms like Render
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc
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
from medical_search import fetch_dynamic_disease_info
from chat import router as chat_router
from intake import router as intake_router

app = FastAPI(title="PocketDoc AI Microservice", version="1.0")

# Enable CORS for Next.js proxy calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-routers
app.include_router(chat_router, prefix="/chat", tags=["Pre-Consultation Chat"])
app.include_router(intake_router, prefix="/intake", tags=["Doctor Handoff"])

# ------------------------------------------------------------------
# 1. LOAD ONLY THE TFLITE MODEL (No .keras model loaded into RAM)
# ------------------------------------------------------------------
TFLITE_MODEL_PATH = Path("models/skin_model.tflite")
print(f"Loading TFLite Model from {TFLITE_MODEL_PATH}...")
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite Model loaded successfully! RAM baseline ~70MB.")


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes to a preprocessed array for EfficientNetB0."""
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
    return {"status": "online", "model": "EfficientNetB0 TFLite (23 classes)"}


@app.post("/predict")
async def predict_skin_disease(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No image file provided")

    input_tensor = None
    try:
        image_bytes = await file.read()
        input_tensor = preprocess_image_bytes(image_bytes)

        # ------------------------------------------------------------------
        # 2. TFLITE INFERENCE (Ultra-fast & lightweight)
        # ------------------------------------------------------------------
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]

        # Extract Top 3 Predictions
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

        # Fetch dynamic medical insight details
        medical_insights = fetch_dynamic_disease_info(top_clean_label)

        response_payload = {
            "success": True,
            "top_prediction": top_3_predictions[0],
            "top_3_predictions": top_3_predictions,
            "triage_level": triage_level,
            "explanation": {
                "heatmap_image": "",  # Skipped to keep RAM under 100MB
                "description": f"AI evaluation identified key features consistent with '{top_clean_label}'.",
                "insights": medical_insights
            }
        }

        return response_payload

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")
    finally:
        if input_tensor is not None:
            del input_tensor
        gc.collect()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)