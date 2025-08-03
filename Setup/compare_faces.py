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

# ==== Load PCA ====
def load_trained_pca(pca_path='./models/pca_512_to_128.pkl'):
    with open(pca_path, 'rb') as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pca_data = pickle.load(f)
    return pca_data['pca_model']

# ==== Load Hyperplanes ====
def load_neuralhash_hyperplanes(dat_path):
    with open(dat_path, 'rb') as f:
        buffer = f.read()
    data = np.frombuffer(buffer, dtype=np.int8)
    num_rows = data.size // 128
    hyperplanes = data.reshape((num_rows, 128)).astype(np.float32)
    return hyperplanes[:96]  # Use only first 96 rows

# ==== Face Embedding ====
def align_and_embed(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    face = mtcnn(img_rgb)
    if face is None:
        raise ValueError(f"No face detected in {img_path}")
    face = face.unsqueeze(0).to(device)
    with torch.no_grad():
        emb_512 = arcface(face).cpu().numpy().squeeze()
    return emb_512

# ==== PCA + Normalize ====
def reduce_embedding(embedding_512, pca):
    emb_128 = pca.transform([embedding_512])[0]
    emb_128 = emb_128 / np.linalg.norm(emb_128)
    return emb_128

# ==== Compute NeuralHash ====
def compute_neuralhash(embedding_128, hyperplanes):
    return (np.dot(hyperplanes, embedding_128) > 0).astype(np.uint8)

# ==== Hamming Distance ====
def hamming_distance(bits1, bits2):
    return np.sum(bits1 != bits2)

# ==== Main Compare Function ====
def compare_faces(img1_path, img2_path, pca, hyperplanes):
    emb1 = align_and_embed(img1_path)
    emb2 = align_and_embed(img2_path)

    emb1_128 = reduce_embedding(emb1, pca)
    emb2_128 = reduce_embedding(emb2, pca)

    hash1 = compute_neuralhash(emb1_128, hyperplanes)
    hash2 = compute_neuralhash(emb2_128, hyperplanes)

    h_dist = hamming_distance(hash1, hash2)

    return {
        'hash1': hash1,
        'hash2': hash2,
        'hamming_distance': h_dist
    }

# ==== Entry Point ====
if __name__ == "__main__":
    image1 = "./data/probe/image4.jpg"
    image2 = "./data/probe/image5.jpg"  # Replace with your second image

    try:
        pca = load_trained_pca("./models/pca_512_to_128.pkl")
        hyperplanes = load_neuralhash_hyperplanes("./models/neuralhash_128x96_seed1.dat")

        result = compare_faces(image1, image2, pca, hyperplanes)

        print("Hamming Distance:", result['hamming_distance'])
        print("Similarity:", "Same Person" if result['hamming_distance'] < 15 else "Different People")

    except Exception as e:
        print("Error:", str(e))
