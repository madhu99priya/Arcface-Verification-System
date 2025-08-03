# psi_match_client_server.py
import json
from neural_hash_gen import (
    load_trained_pca,
    load_neuralhash_hyperplanes,
    process_image
)
from generate_reference_db import (
    generate_private_key,
    blind_neuralhash
)

def hamming_distance(bits1, bits2):
    return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))

def load_reference_db(path="./db/reference_blinded_hashes.json"):
    with open(path, "r") as f:
        return json.load(f)

def check_match_approx(probe_bits, ref_db, threshold=15):
    candidates = []
    for person, entries in ref_db.items():
        for entry in entries:
            ref_bits = entry["hash_bits"]
            dist = hamming_distance(probe_bits, ref_bits)
            if dist <= threshold:
                candidates.append((person, entry["blinded_hash"], dist))
    return candidates

def main():
    probe_image_path = "./data/probe/image19.jpeg"
    pca_path = "./models/pca_512_to_128.pkl"
    dat_path = "./models/neuralhash_128x96_seed1.dat"
    ecc_key_path = "./models/ecc_private.pem"
    ref_db_path = "./db/reference_blinded_hashes.json"

    # Load everything
    pca = load_trained_pca(pca_path)
    hyperplanes = load_neuralhash_hyperplanes(dat_path)
    private_key = generate_private_key(ecc_key_path)
    reference_db = load_reference_db(ref_db_path)

    # Process probe image
    try:
        result = process_image(probe_image_path, pca, hyperplanes)
        probe_bits = result['hash_bits'].tolist()
        probe_hex = result['neural_hash']
        blinded_probe = blind_neuralhash(probe_hex, private_key)

        print(f"\nBlinded Probe Hash: {blinded_probe}")

        matches = check_match_approx(probe_bits, reference_db)

        if matches:
            print("\n✅ Potential Matches:")
            for person, blinded_ref, dist in matches:
                match_status = "✓" if blinded_ref == blinded_probe else "⚠️ hash mismatch"
                print(f" - {person} | Hamming Distance: {dist} | {match_status}")
        else:
            print("\n❌ No Approximate Match Found.")

    except Exception as e:
        print(f"[x] Error: {e}")

if __name__ == "__main__":
    main()
