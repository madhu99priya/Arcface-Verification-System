# from flask import Flask, request, render_template, redirect, url_for
# import os
# import numpy as np
# from werkzeug.utils import secure_filename
# from main import generate_neuralhash, load_pca_model, load_hyperplanes, bits_to_hex

# app = Flask(__name__)
# UPLOAD_FOLDER = 'static/uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# # Load models once
# PCA_MODEL_PATH = './models/pca_512_to_128.pkl'
# HYPERPLANE_PATH = './models/neuralhash_128x96_seed1.dat'
# pca_model = load_pca_model(PCA_MODEL_PATH)
# hyperplanes = load_hyperplanes(HYPERPLANE_PATH)

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/compare', methods=['POST'])
# def compare():
#     if 'image1' not in request.files or 'image2' not in request.files:
#         return 'Missing file input', 400

#     img1 = request.files['image1']
#     img2 = request.files['image2']

#     if img1.filename == '' or img2.filename == '':
#         return 'No selected file', 400

#     filename1 = secure_filename(img1.filename)
#     filename2 = secure_filename(img2.filename)
#     path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
#     path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)

#     img1.save(path1)
#     img2.save(path2)

#     try:
#         bits1 = generate_neuralhash(path1, pca_model, hyperplanes)
#         bits2 = generate_neuralhash(path2, pca_model, hyperplanes)

#         hash1 = bits_to_hex(bits1)
#         hash2 = bits_to_hex(bits2)

#         hamming_dist = int(np.sum(bits1 != bits2))
#         similarity = f"{96 - hamming_dist}/96"

#         return render_template('result.html',
#                                image1=path1,
#                                image2=path2,
#                                hash1=hash1,
#                                hash2=hash2,
#                                hamming=hamming_dist,
#                                similarity=similarity)

#     except Exception as e:
#         return f"Error: {str(e)}", 500

# if __name__ == '__main__':
#     os.makedirs(UPLOAD_FOLDER, exist_ok=True)
#     app.run(debug=True)
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
