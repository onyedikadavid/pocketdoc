# main.py
from src.train import run_training
from src.evaluate import evaluate_saved_model
import config

if __name__ == "__main__":
    print("=== STEP 1: TRAINING ===")
    run_training()

    print("\n=== STEP 2: EVALUATION ===")
    model_file = config.MODEL_SAVE_DIR / "skin_disease_model.keras"
    evaluate_saved_model(model_file)