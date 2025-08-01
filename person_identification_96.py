import os
import numpy as np
from PCA_neural_hash_96 import (
    process_image,
    load_trained_pca,
    load_hyperplanes,
    cosine_similarity_score
)

def identify_person_by_matches(probe_image_path, database_dir, dat_file, similarity_threshold=0.75):
    print(f"\n[INFO] Processing probe image: {probe_image_path}")
    pca = load_trained_pca()
    hyperplanes = load_hyperplanes(dat_file)
    probe_result = process_image(probe_image_path, pca, hyperplanes)

    if not probe_result['success']:
        print(f"[ERROR] Failed to process probe image: {probe_result['error']}")
        return None

    probe_emb_96 = probe_result['embedding_96']
    folder_match_counts = {}

    for person_name in os.listdir(database_dir):
        person_folder = os.path.join(database_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"[INFO] Checking folder: {person_name}")
        match_count = 0
        total_images = 0

        for img_file in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_file)
            result = process_image(img_path, pca, hyperplanes)
            if not result['success']:
                continue

            total_images += 1
            similarity = cosine_similarity_score(probe_emb_96, result['embedding_96'])

            if similarity >= similarity_threshold:
                match_count += 1

        folder_match_counts[person_name] = match_count
        print(f"  → {match_count} / {total_images} images matched in folder '{person_name}'")

    if not folder_match_counts:
        print("\n[ERROR] No valid folders found.")
        return {'match': False}

    # Find the folder with the highest number of matches
    best_match_person = max(folder_match_counts, key=folder_match_counts.get)
    max_matches = folder_match_counts[best_match_person]

    if max_matches > 0:
        print("\n✅ MATCH FOUND!")
        print(f"Matched Person: {best_match_person}")
        print(f"Number of Matches: {max_matches}")
        return {
            'match': True,
            'person': best_match_person,
            'match_count': max_matches
        }
    else:
        print("\n❌ NO MATCH FOUND in database.")
        return {
            'match': False
        }


if __name__ == "__main__":
    
    # Inputs
    probe_image_path = "./data/probe/image15.jpg"
    database_dir = "./data/dataset"
    dat_file = "./models/neuralhash_128x96_seed1.dat"
    similarity_threshold = 0.75

   
    identify_person_by_matches(probe_image_path, database_dir, dat_file, similarity_threshold)
