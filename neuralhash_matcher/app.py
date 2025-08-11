import os
import json
import base64
from flask import Flask, request, render_template, jsonify
from neuralhash_utils import load_pca_model, load_hyperplanes, generate_neuralhash, bits_to_hex
from voucher import create_inner_ciphertext, derive_k_outer_from_hash_hex, create_outer_ciphertext, encode_b64
from hashlib import sha256
from datetime import datetime
import cv2

app = Flask(__name__)

# Config paths
PCA_MODEL_PATH = './models/pca_512_to_128.pkl'
HYPERPLANE_PATH = './models/neuralhash_128x96_seed1.dat'
HASH_DB_PATH = './db/hashes.json'
SHARES_DIR = './db/accounts'
DEVICE_KEYS_DIR = './device_keys'
VOUCHERS_DIR = './db/vouchers'
THRESHOLD_DIST = 15
os.makedirs(VOUCHERS_DIR, exist_ok=True)

# Load models and DB
pca = load_pca_model(PCA_MODEL_PATH)
hyperplanes = load_hyperplanes(HYPERPLANE_PATH)
with open(HASH_DB_PATH, 'r') as f:
    stored = json.load(f)
    stored_hashes = {entry['id']: entry for entry in stored}


def hamming_distance(bits1, bits2):
    return sum(int(b1 != b2) for b1, b2 in zip(bits1, bits2))  # ensure Python int


def save_voucher(account, ref_id, outer_ct_b64, share_str):
    out_file = os.path.join(VOUCHERS_DIR, f'{account}.json')
    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            arr = json.load(f)
    else:
        arr = []
    arr.append({
        'ref_id': ref_id,
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'outer_ct': outer_ct_b64,
        'share': share_str
    })
    with open(out_file, 'w') as f:
        json.dump(arr, f, indent=2)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/identify', methods=['POST'])
def identify():
    try:
        data_url = request.json.get('image')
        account = request.json.get('account', 'test_account')
        if not data_url:
            return jsonify({'error': 'No image data received'}), 400

        header, encoded = data_url.split(',', 1)
        image_data = base64.b64decode(encoded)

        temp_image_path = 'temp_capture.jpg'
        with open(temp_image_path, 'wb') as f:
            f.write(image_data)

        bits = list(generate_neuralhash(temp_image_path, pca, hyperplanes))  # ensure list
        hash_hex = bits_to_hex(bits)

        # Load device key
        device_key_path = os.path.join(DEVICE_KEYS_DIR, f'{account}.key')
        if not os.path.exists(device_key_path):
            return jsonify({'error': 'Device key not found for account; run setup_account_shares.py'}), 500
        with open(device_key_path, 'rb') as f:
            K_inner = f.read()

        neuralhash_bytes = bytes.fromhex(hash_hex)
        inner_ct = create_inner_ciphertext(K_inner, neuralhash_bytes, visual_bytes=b'')

        # Pop a share from server-side share pool
        shares_file = os.path.join(SHARES_DIR, f'{account}_shares.json')
        if not os.path.exists(shares_file):
            return jsonify({'error': 'Shares pool not found; run setup_account_shares.py'}), 500
        with open(shares_file, 'r') as f:
            pool = json.load(f)
        if not pool['shares']:
            return jsonify({'error': 'No more pre-generated shares available for demo'}), 500
        my_share = pool['shares'].pop(0)
        with open(shares_file, 'w') as f:
            json.dump(pool, f, indent=2)

        K_outer = derive_k_outer_from_hash_hex(hash_hex)
        outer_ct = create_outer_ciphertext(K_outer, inner_ct, my_share, metadata=b'')

        # Matching
        matches = []
        for ref_id, entry in stored_hashes.items():
            ref_bits = entry['hash_bits']
            dist = int(hamming_distance(bits, ref_bits))  # force Python int
            if dist <= THRESHOLD_DIST:
                matches.append((ref_id, dist))

        response = {'status': 'no_match', 'matches': []}
        if matches:
            for ref_id, dist in matches:
                save_voucher(account, ref_id, encode_b64(outer_ct), my_share)
                response['matches'].append({'id': ref_id, 'distance': int(dist)})
            response['status'] = 'match'

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/capture', methods=['POST'])
def capture():
    try:
        # Capture frame from webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return jsonify({"error": "Cannot open camera"})
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return jsonify({"error": "Failed to capture image"})

        # Save temp image
        temp_image_path = 'temp_capture.jpg'
        cv2.imwrite(temp_image_path, frame)

        # Generate neural hash
        bits = list(generate_neuralhash(temp_image_path, pca, hyperplanes))
        hash_hex = bits_to_hex(bits)

        # Match with stored hashes
        matches = []
        for ref_id, entry in stored_hashes.items():
            ref_bits = entry['hash_bits']
            dist = hamming_distance(bits, ref_bits)
            if dist <= THRESHOLD_DIST:
                matches.append((ref_id, dist))

        os.remove(temp_image_path)

        # Return just the person's name
        if matches:
            # Take best match (lowest distance)
            best_match_id = min(matches, key=lambda x: x[1])[0]
            # Extract name from filename (before first underscore or dot)
            person_name = best_match_id.split('_')[0].split('.')[0]
            return jsonify({"status": "match", "person": person_name})
        else:
            return jsonify({"status": "no_match", "person": None})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True)
