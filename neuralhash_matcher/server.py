import os
import json
from flask import Flask, request, jsonify
from secretsharing import PlaintextToHexSecretSharer
from neuralhash_utils import bits_to_hex
from datetime import datetime

app = Flask(__name__)

SHARES_DIR = './db/accounts'
THRESHOLD_K = 3

# Load stored hashes (reference database)
with open('./db/hashes.json', 'r') as f:
    stored = json.load(f)
    stored_hashes = {entry['id']: entry for entry in stored}

# Helper: Hamming distance
def hamming_distance(bits1, bits2):
    return sum(int(b1 != b2) for b1, b2 in zip(bits1, bits2))

# Keep shares per account in memory (or you can persist in files)
shares_pool = {}

# Only print and check distances for these Madhusha images
DEBUG_IDS = [
    "Madhusha_frame_4.jpg",
    "Madhusha_frame_5.jpg",
    "Madhusha_frame_6.jpg",
    "Madhusha_frame_7.jpg",
    "Madhusha_Madhusha1.jpg",
    "Madhusha_Madhusha2.jpg"

]

@app.route('/submit_share', methods=['POST'])
def submit_share():
    data = request.json
    account = data.get('account')
    share = data.get('share')

    if not account or not share:
        return jsonify({'error': 'Missing account or share'}), 400

    if account not in shares_pool:
        shares_pool[account] = []

    shares_pool[account].append(share)

    if len(shares_pool[account]) >= THRESHOLD_K:
        try:
            reconstructed_hex = PlaintextToHexSecretSharer.recover_secret(shares_pool[account])
            print(f"[DEBUG] Reconstructed hex: {reconstructed_hex}")

            bits = [int(x) for x in bin(int(reconstructed_hex, 16))[2:].zfill(96)] 

            matches = []
            THRESHOLD_DIST = 20

            for ref_id, entry in stored_hashes.items():
                ref_bits = entry['hash_bits']
                dist = hamming_distance(bits, ref_bits)

                # Only print the distance for the DEBUG IDs
                if ref_id in DEBUG_IDS:
                    print(f"[DEBUG] Comparing with {ref_id}: distance = {dist}")

                # Check match on ALL ids in the database
                if dist <= THRESHOLD_DIST:
                    matches.append({'id': ref_id, 'distance': dist})

            shares_pool[account] = []  # reset for next sequence

            if matches:
                print(f"[DEBUG] Matches found: {matches}")
                return jsonify({'status': 'match', 'matches': matches})
            else:
                print("[DEBUG] No matches found.")
                return jsonify({'status': 'no_match'})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

    else:
        return jsonify({'status': 'waiting', 'shares_received': len(shares_pool[account])})

if __name__ == '__main__':
    app.run(debug=True)
