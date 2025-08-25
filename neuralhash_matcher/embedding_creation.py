import os
import numpy as np
import random
import json
from tqdm import tqdm
from neuralhash_utils import load_pca_model, load_hyperplanes, generate_neuralhash

# ===== Load model =====
pca = load_pca_model('../models/pca_512_to_128.pkl')
hyperplanes = load_hyperplanes('../models/neuralhash_128x96_seed1.dat')

# ===== Define Paths =====
dataset_path = "C:\\Users\\ASUS\\Desktop\\nFilterd"
# --- New: Define the path for the new folder of cropped images ---
cropped_faces_path = "C:\\Users\\ASUS\\Desktop\\nFilterd_cropped_faces"
# Create the main output directory if it doesn't exist
os.makedirs(cropped_faces_path, exist_ok=True)
# -----------------------------------------------------------------

reduced_data = {}

def get_embedding_and_save_crop(image_path, cropped_save_path):
    """
    Generate NeuralHash (binary string) for an image and save the cropped face.
    """
    bits = generate_neuralhash(image_path, pca, hyperplanes, cropped_image_save_path=cropped_save_path)
    if bits is None:
        return None
    # Convert to string "010101..."
    return ''.join(str(int(b)) for b in bits)

# ===== Iterate over dataset =====
folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

for folder in tqdm(folders, desc="Processing Folders", ncols=100, unit="folder"):
    folder_path = os.path.join(dataset_path, folder)
    
    # --- New: Create a corresponding subfolder in the output directory ---
    output_folder_path = os.path.join(cropped_faces_path, folder)
    os.makedirs(output_folder_path, exist_ok=True)
    # --------------------------------------------------------------------

    all_images = sorted(os.listdir(folder_path))
    selected_images = random.sample(all_images, min(50, len(all_images)))
    reduced_data[folder] = []

    for img_file in tqdm(selected_images, desc=f"Images in {folder}", leave=False, ncols=100, unit="img"):
        img_path = os.path.join(folder_path, img_file)
        
        # --- New: Define the full save path for the cropped image ---
        cropped_save_path = os.path.join(output_folder_path, img_file)
        # -------------------------------------------------------------
        
        try:
            # Updated function call to save the cropped image
            h = get_embedding_and_save_crop(img_path, cropped_save_path)
            if h is not None:
                reduced_data[folder].append({
                    "filename": img_file,
                    "hash": h
                })
        except ValueError as e:
            # Skip images with no face detected
            print(f"⚠️ Skipped {img_file} in {folder} ({e})")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"❌ Error processing {img_file} in {folder}: {e}")

# ===== Save JSON =====
with open("reduced_dataset2.json", "w") as f:
    json.dump(reduced_data, f, indent=2)

print(f"✅ Done! JSON saved as 'reduced_dataset.json'. Cropped faces saved in '{cropped_faces_path}'.")