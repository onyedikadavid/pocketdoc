# src/preprocessing.py
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
import config

# Capital 'D' to match your exact directory casing
PREPROCESSED_DATA_DIR = Path("Data/preprocessed")

def create_data_generators():
    """Loads preprocessed images directly from Data/preprocessed/ using validation split."""
    
    # Train generator with data augmentation and validation split
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=config.VALIDATION_SPLIT  # Uses your config split ratio
    )

    # Pure loader generator for test set (no augmentation/split)
    test_datagen = ImageDataGenerator()

    # 1. Training generator (reads subset='training' from train folder)
    train_gen = train_datagen.flow_from_directory(
        PREPROCESSED_DATA_DIR / "train",
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    # 2. Validation generator (reads subset='validation' from train folder)
    val_gen = train_datagen.flow_from_directory(
        PREPROCESSED_DATA_DIR / "train",
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # 3. Test generator (reads from test folder)
    test_gen = test_datagen.flow_from_directory(
        PREPROCESSED_DATA_DIR / "test",
        target_size=config.IMAGE_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_gen, val_gen, test_gen

def compute_balanced_weights(train_gen):
    class_labels = train_gen.classes
    weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(class_labels),
        y=class_labels
    )
    return dict(enumerate(weights))