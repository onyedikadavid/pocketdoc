# config.py
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "preprocessed"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
MODEL_SAVE_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

# Hyperparameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5  # (Smaller learning rate for Phase 2)
VALIDATION_SPLIT = 0.2