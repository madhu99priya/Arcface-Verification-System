# File: 2_crop_faces_best_100.py
# Purpose: For each person, finds the 100 best images with a single, high-confidence face and saves the crops.

import os
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN
import torch

# --- Configuration ---
FILTERED_DATASET_DIR = "C:\\Users\\ASUS\\Desktop\\filteredd"
CROPPED_DATASET_DIR = "C:\\Users\\ASUS\\Desktop\\cropped_best_100"  # New folder
MAX_IMAGES_PER_PERSON = 100


# --- Main Logic ---
def main():
    """
    Analyzes all images, ranks them by face detection quality,
    and saves the top 100 per person.
    """
    print("=" * 60)
    print("PHASE 2: SELECTIVE CROPPING (BEST 100 FACES PER PERSON)")
    print("=" * 60)

    # --- Load Model ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')

    # keep_all=True is needed to count the faces accurately.
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)

    os.makedirs(CROPPED_DATASET_DIR, exist_ok=True)

    print(f"\nAnalyzing images from: {FILTERED_DATASET_DIR}")
    print(f"Saving best {MAX_IMAGES_PER_PERSON} crops per person to: {CROPPED_DATASET_DIR}")

    persons = sorted(os.listdir(FILTERED_DATASET_DIR))

    for person in tqdm(persons, desc="Processing Persons"):
        filtered_person_dir = os.path.join(FILTERED_DATASET_DIR, person)
        cropped_person_dir = os.path.join(CROPPED_DATASET_DIR, person)

        if not os.path.isdir(filtered_person_dir):
            continue

        # --- Pass 1: Analyze and Rank all images for the current person ---
        good_candidates = []

        for img_file in os.listdir(filtered_person_dir):
            img_path = os.path.join(filtered_person_dir, img_file)
            try:
                img = Image.open(img_path).convert('RGB')
                # We need both boxes and probabilities to rank them
                boxes, probs = mtcnn.detect(img)

                # Check for the ideal condition: exactly one face found with high confidence
                if boxes is not None and len(boxes) == 1:
                    confidence = probs[0]
                    good_candidates.append({'path': img_path, 'confidence': confidence})
            except Exception:
                # Silently skip corrupted images during the analysis phase
                continue

        # --- Pass 2: Sort, select the best, and save them ---
        if not good_candidates:
            print(f"\nWarning: No good single-face images found for {person}. Skipping.")
            continue

        # Sort candidates by detection confidence, highest first
        sorted_candidates = sorted(good_candidates, key=lambda x: x['confidence'], reverse=True)

        # Select the top N candidates
        best_candidates = sorted_candidates[:MAX_IMAGES_PER_PERSON]

        # Now, create the directory and save the crops for the best images
        os.makedirs(cropped_person_dir, exist_ok=True)

        for i, candidate in enumerate(best_candidates):
            try:
                img = Image.open(candidate['path']).convert('RGB')
                save_path = os.path.join(cropped_person_dir, f"{i}.jpg")

                # We only need the first face since we know there's only one
                mtcnn(img, save_path=save_path)
            except Exception as e:
                print(f"\nError while saving crop for {candidate['path']}: {e}")

    print("\n✅ Selective cropping of best 100 faces complete.")


if __name__ == "__main__":
    main()
