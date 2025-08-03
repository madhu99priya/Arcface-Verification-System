# generate_reference_db.py
import os
import json
import hashlib
from neural_hash_gen import (
    load_trained_pca,
    load_neuralhash_hyperplanes,
    process_image
)
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization

SECP256R1_ORDER = int("ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551", 16)

def generate_private_key(path="./models/ecc_private.pem"):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(f.read(), password=None)
    private_key = ec.generate_private_key(ec.SECP256R1())
    with open(path, "wb") as f:
        f.write(private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ))
    return private_key

def blind_neuralhash(neural_hash_hex, private_key):
    digest = hashlib.sha256(bytes.fromhex(neural_hash_hex)).digest()
    scalar = int.from_bytes(digest, 'big') % SECP256R1_ORDER
    pub_key_x = private_key.public_key().public_numbers().x
    blinded = (pub_key_x * scalar) % SECP256R1_ORDER
    return hex(blinded)[2:]

def generate_reference_db():
    REFERENCE_ROOT = "./data/dataset"
    OUT_JSON = "./db/reference_blinded_hashes.json"
    PCA_PATH = "./models/pca_512_to_128.pkl"
    DAT_PATH = "./models/neuralhash_128x96_seed1.dat"
    ECC_KEY_PATH = "./models/ecc_private.pem"

    os.makedirs("./db", exist_ok=True)
    pca = load_trained_pca(PCA_PATH)
    hyperplanes = load_neuralhash_hyperplanes(DAT_PATH)
    private_key = generate_private_key(ECC_KEY_PATH)

    ref_db = {}

    for person_folder in os.listdir(REFERENCE_ROOT):
        person_path = os.path.join(REFERENCE_ROOT, person_folder)
        if not os.path.isdir(person_path):
            continue

        images = [f for f in os.listdir(person_path)
                  if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        if not images:
            print(f"[x] No valid images found for {person_folder}")
            continue

        ref_db[person_folder] = []

        for image_file in images:
            image_path = os.path.join(person_path, image_file)

            try:
                result = process_image(image_path, pca, hyperplanes)
                neural_hash = result['neural_hash']
                blinded = blind_neuralhash(neural_hash, private_key)
                ref_db[person_folder].append({
                    "hash_bits": result['hash_bits'].tolist(),
                    "blinded_hash": blinded
                })
                print(f"[✓] {person_folder} - {image_file}")
            except Exception as e:
                print(f"[x] Failed {person_folder}/{image_file}: {e}")

    with open(OUT_JSON, "w") as f:
        json.dump(ref_db, f, indent=2)

    print(f"\n✅ Saved blinded reference DB: {OUT_JSON}")

if __name__ == "__main__":
    generate_reference_db()
