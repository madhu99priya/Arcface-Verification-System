# voucher_generator.py

import json
import time
import secrets
import os
import random
from hashlib import sha256
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad
from Crypto.Protocol.SecretSharing import Shamir

from generate_reference_db import generate_private_key, blind_neuralhash
from neural_hash_gen import process_image, load_trained_pca, load_neuralhash_hyperplanes

def generate_random_key():
    return secrets.token_bytes(16)  # AES-128 (128 bits = 16 bytes)

def aes_encrypt(key, data_dict):
    cipher = AES.new(key, AES.MODE_CBC)
    ct_bytes = cipher.encrypt(pad(json.dumps(data_dict).encode(), AES.block_size))
    return {
        "iv": cipher.iv.hex(),
        "ciphertext": ct_bytes.hex()
    }

def generate_voucher(image_path, camera_id="CAM01", threshold=3, total_shares=5):
    # Load models
    pca = load_trained_pca("./models/pca_512_to_128.pkl")
    hyperplanes = load_neuralhash_hyperplanes("./models/neuralhash_128x96_seed1.dat")
    private_key = generate_private_key("./models/ecc_private.pem")

    # Generate NeuralHash
    result = process_image(image_path, pca, hyperplanes)
    neural_hash = result["neural_hash"]

    # Prepare metadata
    metadata = {
        "timestamp": time.time(),
        "camera_id": camera_id,
        "neural_hash": neural_hash
    }

    # Generate a fresh random AES key
    inner_key = generate_random_key()
    key_int = int.from_bytes(inner_key, byteorder='big')
    print(f"[DEBUG] AES key (int): {key_int}")

    # Split the key using Shamir Secret Sharing
    shares = Shamir.split(threshold, total_shares, key_int)
    x_values = [x for x, _ in shares]
    print(f"[DEBUG] Shamir x-values available: {x_values}")

    # Randomly pick a unique share to include in this voucher
    selected_share = random.choice(shares)
    x, y = selected_share

    if isinstance(y, bytes):
        y_hex = y.hex()
    else:
        y_hex = hex(y)

    # Encrypt metadata with AES key
    inner_encrypted = aes_encrypt(inner_key, metadata)

    # ECC-blind the NeuralHash
    blinded_hash = blind_neuralhash(neural_hash, private_key)

    # Final voucher structure
    voucher = {
        "blinded_hash": blinded_hash,
        "header": sha256(bytes.fromhex(neural_hash)).hexdigest(),
        "payload": {
            "share": {
                "x": x,
                "y": y_hex
            },
            "inner_encrypted": inner_encrypted
        }
    }

    return voucher

def get_next_voucher_filename():
    os.makedirs("./vouchers", exist_ok=True)
    counter = 1
    while os.path.exists(f"./vouchers/voucher{counter}.json"):
        counter += 1
    return f"./vouchers/voucher{counter}.json"

if __name__ == "__main__":
    image_path = "./data/probe/image13.jpg"
    voucher = generate_voucher(image_path)

    filename = get_next_voucher_filename()
    with open(filename, "w") as f:
        json.dump(voucher, f, indent=2)
    print(f"✅ Voucher generated and saved as: {filename}")
