# src/evaluate.py
import os
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
import config
from src.preprocessing import create_data_generators

def save_predictions_to_csv(model, test_gen, output_dir="outputs", file_name="predictions.csv"):
    """Runs inference on test data and saves image paths, true labels, predicted labels, and confidence scores."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, file_name)

    print("\n--- Generating test predictions for predictions.csv ---")
    
    # 1. Reset generator to start from the first image
    test_gen.reset()

    # 2. Predict raw probabilities for all test samples
    predictions = model.predict(test_gen, verbose=1)
    
    # 3. Get predicted class index and confidence score
    predicted_class_indices = np.argmax(predictions, axis=1)
    confidence_scores = np.max(predictions, axis=1)

    # 4. Map class indices back to actual class label names
    class_labels = {v: k for k, v in test_gen.class_indices.items()}
    predicted_labels = [class_labels[idx] for idx in predicted_class_indices]
    
    # True class labels from generator
    true_class_indices = test_gen.classes
    true_labels = [class_labels[idx] for idx in true_class_indices]

    # File paths of test images
    file_paths = test_gen.filepaths

    # 5. Build DataFrame and export to CSV
    df = pd.DataFrame({
        "image_path": file_paths,
        "true_label": true_labels,
        "predicted_label": predicted_labels,
        "confidence": np.round(confidence_scores, 4)
    })

    df.to_csv(output_path, index=False)
    print(f"Predictions successfully saved to {output_path}")

def evaluate_saved_model(model_path):
    _, _, test_gen = create_data_generators()
    
    if not os.path.exists(model_path):
        print(f"Error: Saved model not found at {model_path}")
        return

    model = tf.keras.models.load_model(model_path)

    print("Evaluating Model on Test Data...")
    results = model.evaluate(test_gen)
    print(f"Test Loss: {results[0]:.4f} - Test Accuracy: {results[1]:.4f}")

    # Generate classification report
    test_gen.reset()
    predictions = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_gen.classes

    class_names = list(test_gen.class_indices.keys())
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save detailed predictions to CSV
    save_predictions_to_csv(model, test_gen, output_dir="outputs", file_name="predictions.csv")

if __name__ == "__main__":
    model_file = config.MODEL_SAVE_DIR / "skin_disease_model.keras"
    evaluate_saved_model(model_file)