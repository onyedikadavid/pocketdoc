import os
import zipfile

def extract_local_zip(zip_name: str, target_dir: str):
    zip_path = os.path.join(target_dir, zip_name)
    
    if not os.path.exists(zip_path):
        print(f"Error: Could not find {zip_path}")
        return

    print(f"Extracting {zip_name} into {target_dir}...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    print("Extraction complete!")
    
    # Optional: Remove zip file after extracting to save disk space
    # os.remove(zip_path)

if __name__ == "__main__":
    RAW_DATA_DIR = "./Data/Raw_data"
    ZIP_FILENAME = "archive (2).zip"  # Matches the file name in your VS Code sidebar
    
    extract_local_zip(ZIP_FILENAME, RAW_DATA_DIR)