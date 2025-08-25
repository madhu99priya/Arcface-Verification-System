# File: calibrate_blur_threshold.py (Corrected)
# Purpose: To analyze sharpness scores from a single folder of images
#          to help you choose the best blur threshold.

import cv2
import os
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# PASTE THE FULL PATH TO YOUR FOLDER OF ~300 IMAGES HERE
SAMPLE_FOLDER_PATH = "C:\\Users\\ASUS\\Desktop\\VGGFace_Dataset\\n000003"
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

NUM_SAMPLES_TO_SHOW = 20 # We will show the 20 blurriest and 20 sharpest

def get_blur_score(image_path):
    """Calculates a blurriness score using the variance of the Laplacian."""
    image = cv2.imread(image_path)
    if image is None: 
        return 0.0, "Unreadable"
    
    # --- THIS IS THE CORRECTED LINE ---
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score, "OK"

# --- Main Logic ---
if not os.path.isdir(SAMPLE_FOLDER_PATH):
    print(f"❌ Error: The folder was not found at the path you provided.")
    print(f"Please check the path: {SAMPLE_FOLDER_PATH}")
    exit()

image_files = [f for f in os.listdir(SAMPLE_FOLDER_PATH) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not image_files:
    print(f"❌ Error: No image files (.jpg, .png) were found in the folder.")
    exit()

# 1. Calculate scores for all images first
print(f"🔍 Analyzing {len(image_files)} images from the folder...")
all_scores = []
for img_file in tqdm(image_files, desc="Calculating Scores"):
    img_path = os.path.join(SAMPLE_FOLDER_PATH, img_file)
    score, _ = get_blur_score(img_path)
    all_scores.append((score, img_file))

# 2. Sort the images by score (lowest to highest)
all_scores.sort(key=lambda item: item[0])

# 3. Print a statistical summary
scores_only = [s[0] for s in all_scores]
print("\n" + "="*50)
print("STATISTICAL SUMMARY OF SHARPNESS SCORES")
print("="*50)
print(f"Lowest Score (Blurriest): {min(scores_only):.2f}")
print(f"Highest Score (Sharpest): {max(scores_only):.2f}")
print(f"Average Score:            {np.mean(scores_only):.2f}")
print("="*50)

# 4. Show the blurriest images
print(f"\n--- 📉 The {NUM_SAMPLES_TO_SHOW} Blurriest Images (Lowest Scores) ---")
for score, name in all_scores[:NUM_SAMPLES_TO_SHOW]:
    print(f"Score: {score:<10.2f} | File: {name}")

# 5. Show the sharpest images
print(f"\n--- 📈 The {NUM_SAMPLES_TO_SHOW} Sharpest Images (Highest Scores) ---")
for score, name in all_scores[-NUM_SAMPLES_TO_SHOW:]:
    print(f"Score: {score:<10.2f} | File: {name}")

print("\n" + "="*50)
print("DECISION TIME")
print("="*50)
print("1. Open your image folder and look at the files listed above.")
print("2. Compare the blurry images (e.g., scores from 10-80) with the clear ones (e.g., scores > 150).")
print("3. Decide on a cutoff point. A good starting point is often around 100.")
print("4. Set the `BLUR_THRESHOLD` in your `1_filter_images.py` script to this value.")
print("="*50)