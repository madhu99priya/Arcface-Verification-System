import os
import numpy as np
import random
import json
from tqdm import tqdm
from neuralhash_utils import load_pca_model, load_hyperplanes, generate_neuralhash

# ===== Load model =====
pca = load_pca_model('../models/pca_512_to_128.pkl')
hyperplanes = load_hyperplanes('../models/neuralhash_128x96_seed1.dat')

dataset_path = r"C:\Users\ASUS\Desktop\VGGFace_Dataset"
reduced_data = {}

def get_embedding(image_path):
    """Generate NeuralHash (binary string) for an image."""
    bits = generate_neuralhash(image_path, pca, hyperplanes)
    if bits is None:
        return None
    # Convert to string "010101..."
    return ''.join(str(int(b)) for b in bits)

# ===== Iterate over dataset =====
folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

for folder in tqdm(folders, desc="Processing Folders", ncols=100, unit="folder"):
    folder_path = os.path.join(dataset_path, folder)
    all_images = sorted(os.listdir(folder_path))
    selected_images = random.sample(all_images, min(50, len(all_images)))  # pick up to 50 images
    reduced_data[folder] = []

    for img_file in tqdm(selected_images, desc=f"Images in {folder}", leave=False, ncols=100, unit="img"):
        img_path = os.path.join(folder_path, img_file)
        try:
            h = get_embedding(img_path)
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
with open("reduced_dataset.json", "w") as f:
    json.dump(reduced_data, f, indent=2)

print("✅ Done! JSON saved as 'reduced_dataset.json'.")
