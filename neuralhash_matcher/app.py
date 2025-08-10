#app.py
from flask import Flask, request, render_template, jsonify
import os
import base64
import json
from main import generate_neuralhash, load_pca_model, load_hyperplanes
from matcher import match_probe_hash

app = Flask(__name__)

# Load models once
PCA_MODEL_PATH = './models/pca_512_to_128.pkl'
HYPERPLANE_PATH = './models/neuralhash_128x96_seed1.dat'
pca_model = load_pca_model(PCA_MODEL_PATH)
hyperplanes = load_hyperplanes(HYPERPLANE_PATH)

# Load and transform hash database
HASH_DB_PATH = './db/hashes.json'
with open(HASH_DB_PATH, 'r') as f:
    raw_entries = json.load(f)
    stored_hashes = {
        f"{entry['person']}_{entry['image']}": entry["hash_bits"]
        for entry in raw_entries
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify', methods=['POST'])
def identify():
    try:
        data_url = request.json.get('image')
        if not data_url:
            return jsonify({'error': 'No image data received'}), 400

        # Decode base64 image
        header, encoded = data_url.split(',', 1)
        image_data = base64.b64decode(encoded)

        # Save temporarily
        temp_image_path = 'temp_capture.jpg'
        with open(temp_image_path, 'wb') as f:
            f.write(image_data)

        # Generate hash
        bits = generate_neuralhash(temp_image_path, pca_model, hyperplanes)

        # Compare with stored hashes
        matches = match_probe_hash(bits, stored_hashes, threshold=15)

        if matches:
            result = {
                'status': 'match',
                'matches': [
                    {'name': name, 'distance': int(dist)} for name, dist in matches
                ]
            }
        else:
            result = {'status': 'no_match'}

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
