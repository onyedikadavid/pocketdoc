# src/preprocess_offline.py
import os
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import config

# Exact paths matching your VS Code explorer structure
RAW_DATA_DIR = Path("Data/Raw_data")
PREPROCESSED_DATA_DIR = Path("Data/preprocessed")

def process_and_save_dataset():
    """Reads raw dataset from Data/Raw_data, resizes images, and saves them to Data/preprocessed."""
    print("--- Starting Offline Preprocessing ---")
    print(f"Reading from: {RAW_DATA_DIR.resolve()}")
    print(f"Saving to:    {PREPROCESSED_DATA_DIR.resolve()}\n")

    if not RAW_DATA_DIR.exists():
        print(f"Error: Could not find raw data directory at {RAW_DATA_DIR}")
        return

    splits = ['train', 'test']

    for split in splits:
        split_src = RAW_DATA_DIR / split
        split_dst = PREPROCESSED_DATA_DIR / split

        if not split_src.exists():
            print(f"Skipping '{split}' directory (not found at {split_src})...")
            continue

        print(f"Processing '{split}' folder...")

        for class_folder in split_src.iterdir():
            if class_folder.is_dir():
                target_folder = split_dst / class_folder.name
                os.makedirs(target_folder, exist_ok=True)

                image_files = list(class_folder.glob("*.*"))
                print(f"  -> Class '{class_folder.name}': processing {len(image_files)} images...")

                for img_path in image_files:
                    try:
                        # Load & resize image cleanly to config resolution (224x224)
                        img = load_img(img_path, target_size=config.IMAGE_SIZE)
                        
                        # Save directly into Data/preprocessed/<split>/<class>/
                        save_target_path = target_folder / img_path.name
                        img.save(save_target_path)
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

    print("\n--- Offline Preprocessing Complete! ---")
    print(f"Processed images saved in: {PREPROCESSED_DATA_DIR.resolve()}")

if __name__ == "__main__":
    process_and_save_dataset()