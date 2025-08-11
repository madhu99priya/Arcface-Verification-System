# preprocess_dataset.py
import os
import json
from neuralhash_utils import load_pca_model, load_hyperplanes, generate_neuralhash, bits_to_hex

PCA_PATH = './models/pca_512_to_128.pkl'
HYPERPLANE_PATH = './models/neuralhash_128x96_seed1.dat'
DATASET_DIR = './data/dataset'
OUTPUT_JSON = './db/hashes.json'

os.makedirs('db', exist_ok=True)

pca = load_pca_model(PCA_PATH)
hyperplanes = load_hyperplanes(HYPERPLANE_PATH)

database = []

for person in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person)
    if not os.path.isdir(person_dir):
        continue
    for img_file in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_file)
        try:
            bits = generate_neuralhash(img_path, pca, hyperplanes)
            database.append({
                'id': f"{person}_{img_file}",
                'person': person,
                'image': img_file,
                'hash_bits': bits.tolist(),
                'hash_hex': bits_to_hex(bits)
            })
        except Exception as e:
            print(f"[x] Error processing {img_path}: {e}")

with open(OUTPUT_JSON, 'w') as f:
    json.dump(database, f, indent=2)

print(f"✅ Saved {len(database)} hashes to {OUTPUT_JSON}")