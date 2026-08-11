# api_server.py
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from class_mappings import CLASS_NAMES, CLEAN_LABELS, determine_triage_level

app = FastAPI(title="PocketDoc AI Microservice", version="1.0")

# Enable CORS for Next.js proxy calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Trained Model
MODEL_PATH = Path("models/skin_disease_model.keras")
print(f"Loading Keras Model from {MODEL_PATH}...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decodes raw image bytes to a preprocessed tensor for EfficientNetB0."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize to EfficientNet input shape (224, 224)
        img = img.resize((224, 224))

        img_array = np.array(img, dtype=np.float32)
        # Add batch dimension -> (1, 224, 224, 3)
        img_array = np.expand_dims(img_array, axis=0)

        # Rescale if model was trained with 1./255 scaling
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
            confidence_score = float(predictions[idx])  # cast off numpy.float32

            top_3_predictions.append({
                "class_name": clean_label,
                "confidence": round(confidence_score, 4)
            })

        # Top prediction details for triage determination
        top_raw_class = CLASS_NAMES[int(top_3_indices[0])]  # cast off numpy.int64
        top_confidence = float(predictions[top_3_indices[0]])

        triage_level = determine_triage_level(top_raw_class, top_confidence)

        # Return exact JSON shape expected by src/app/api/scan/route.ts
        return {
            "success": True,
            "top_prediction": top_3_predictions[0],
            "top_3_predictions": top_3_predictions,
            "triage_level": triage_level,
        }

    except ValueError as e:
        # Bad/corrupt image payload
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)