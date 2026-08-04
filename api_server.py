# api_server.py
import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

class ScanRequest(BaseModel):
    image: str  # Base64 encoded string from Next.js

def preprocess_base64_image(base64_string: str) -> np.ndarray:
    """Decodes base64 string to PIL Image and preprocesses for EfficientNetB0."""
    try:
        # Strip header if present (e.g., "data:image/jpeg;base64,")
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]

        image_bytes = base64.b64decode(base64_string)
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
def predict_skin_disease(payload: ScanRequest):
    if not payload.image:
        raise HTTPException(status_code=400, detail="No image data provided")

    try:
        # Preprocess Image
        input_tensor = preprocess_base64_image(payload.image)

        # Run Prediction Inference
        predictions = model.predict(input_tensor, verbose=0)[0]

        # Get Top-3 Predictions
        top_3_indices = np.argsort(predictions)[-3:][::-1]

        formatted_predictions = []
        for idx in top_3_indices:
            raw_class_name = CLASS_NAMES[idx]
            clean_label = CLEAN_LABELS.get(raw_class_name, raw_class_name)
            confidence_score = float(predictions[idx])

            formatted_predictions.append({
                "label": clean_label,
                "confidence": round(confidence_score, 4)
            })

        # Top prediction details for triage determination
        top_raw_class = CLASS_NAMES[top_3_indices[0]]
        top_confidence = float(predictions[top_3_indices[0]])

        triage_level = determine_triage_level(top_raw_class, top_confidence)

        # Return exact JSON shape expected by src/app/api/scan/route.ts
        return {
            "predictions": formatted_predictions,
            "triageLevel": triage_level
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Runs FastAPI server on http://localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)