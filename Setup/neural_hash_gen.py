#neural_hash_gen.py

import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.decomposition import PCA
import pickle
import warnings

# ==== Load Models ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, device=device)
arcface = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==== Load PCA (512 -> 128) ====
def load_trained_pca(pca_path='./models/pca_512_to_128.pkl'):
    with open(pca_path, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress version mismatch warnings
            pca_data = pickle.load(f)
    return pca_data['pca_model']

# ==== Load Apple NeuralHash Hyperplanes (96 x 128 from 384 x 128 int8) ====
def load_neuralhash_hyperplanes(dat_path):
    with open(dat_path, 'rb') as f:
        buffer = f.read()

    data = np.frombuffer(buffer, dtype=np.int8)

    if data.size % 128 != 0:
        raise ValueError("Invalid projection matrix shape: not divisible by 128")

    num_rows = data.size // 128
    print(f"Loaded NeuralHash matrix: {num_rows} x 128")

    hyperplanes = data.reshape((num_rows, 128)).astype(np.float32)

    # Pick first 96 hyperplanes for single hash seed
    return hyperplanes[:96]

# ==== Face Alignment and Embedding (512D) ====
def align_and_embed(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)
    if face is None:
        raise ValueError("No face detected in image.")

    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb_512 = arcface(face).cpu().numpy().squeeze()

    return emb_512  # shape: (512,)

# ==== Project to 128D using PCA ====
def apply_pca(embedding_512, pca):
    embedding_128 = pca.transform([embedding_512])[0]
    return embedding_128

# ==== Generate 96-bit NeuralHash ====
def compute_neuralhash_96bits(embedding_128, hyperplanes):
    embedding_128 = embedding_128 / np.linalg.norm(embedding_128)
    bits = (np.dot(hyperplanes, embedding_128) > 0).astype(np.uint8)
    return bits  # shape: (96,)

# ==== Convert Bit Array to Hex String ====
def bits_to_hex(bits):
    return ''.join(f'{int("".join(map(str, bits[i:i+8])), 2):02x}' for i in range(0, len(bits), 8))

# ==== Full Image-to-Hash Pipeline ====
def process_image(img_path, pca, hyperplanes):
    emb_512 = align_and_embed(img_path)
    emb_128 = apply_pca(emb_512, pca)
    hash_bits = compute_neuralhash_96bits(emb_128, hyperplanes)
    neural_hash = bits_to_hex(hash_bits)
    return {
        'embedding_128': emb_128,
        'hash_bits': hash_bits,
        'neural_hash': neural_hash
    }

# ==== Main Entry Point ====
if __name__ == "__main__":
    image_path = "./data/probe/image8.jpg"
    pca_path = "./models/pca_512_to_128.pkl"
    dat_path = "./models/neuralhash_128x96_seed1.dat"

    try:
        pca = load_trained_pca(pca_path)
        hyperplanes = load_neuralhash_hyperplanes(dat_path)

        result = process_image(image_path, pca, hyperplanes)
        print("Generated NeuralHash:", result['neural_hash'])
    except Exception as e:
        print("Error:", str(e))
