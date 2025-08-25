# File: 1_filter_images_with_resolution_check.py
import cv2
import os
from tqdm import tqdm
import shutil

# --- Configuration ---
RAW_DATASET_DIR = "C:\\Users\\ASUS\\Desktop\\VGGFace_Dataset"
FILTERED_DATASET_DIR = "C:\\Users\\ASUS\\Desktop\\nFilterd"

# --- MODIFIED: Added a minimum resolution setting ---
BLUR_THRESHOLD = 800.0
MIN_RESOLUTION = (160, 160) # Width, Height

def get_image_stats(image_path):
    """
    Calculates sharpness score and gets image dimensions.
    Returns (score, width, height).
    """
    image = cv2.imread(image_path)
    if image is None: 
        return 0.0, 0, 0 # Return zeros for unreadable images
    
    # Get dimensions
    height, width, _ = image.shape
    
    # Calculate blur score
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return score, width, height

# --- Main Logic ---
if os.path.exists(FILTERED_DATASET_DIR):
    print(f"Filtered dataset already exists at {FILTERED_DATASET_DIR}. Skipping.")
else:
    os.makedirs(FILTERED_DATASET_DIR)
    print(f"Filtering images from {RAW_DATASET_DIR}...")
    
    total_images_processed = 0
    images_kept = 0

    for person in tqdm(os.listdir(RAW_DATASET_DIR), desc="Processing Persons"):
        raw_person_dir = os.path.join(RAW_DATASET_DIR, person)
        filtered_person_dir = os.path.join(FILTERED_DATASET_DIR, person)
        
        if not os.path.isdir(raw_person_dir): continue
        os.makedirs(filtered_person_dir, exist_ok=True)
        
        for img_file in os.listdir(raw_person_dir):
            total_images_processed += 1
            img_path = os.path.join(raw_person_dir, img_file)
            
            # --- MODIFIED: Get all stats in one go for efficiency ---
            blur_score, width, height = get_image_stats(img_path)
            
            # --- MODIFIED: Combined quality check ---
            is_sharp_enough = blur_score > BLUR_THRESHOLD
            is_large_enough = width >= MIN_RESOLUTION[0] and height >= MIN_RESOLUTION[1]

            if is_sharp_enough and is_large_enough:
                shutil.copy2(img_path, os.path.join(filtered_person_dir, img_file))
                images_kept += 1

    print("\n✅ Filtering complete.")
    print(f"   - Total images processed: {total_images_processed}")
    print(f"   - Images kept (passing filters): {images_kept}")
    print(f"   - Images discarded: {total_images_processed - images_kept}")