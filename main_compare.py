import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import warnings

# Suppress the sklearn version warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load models
print("Loading face detection and recognition models...")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
print("Models loaded successfully!")

# Load and align face
def load_face(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not load image: {img_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)
    if face is None:
        raise ValueError(f"No face detected in: {img_path}")
    return face.unsqueeze(0).to(device)

# Get 512D embedding
def get_embedding(face_tensor):
    with torch.no_grad():
        emb = model(face_tensor)
    return emb.cpu().numpy().squeeze()

# Load PCA (512 → 128)
def load_pca(pca_path):
    try:
        with open(pca_path, 'rb') as f:
            pca_data = pickle.load(f)
            if isinstance(pca_data, dict) and 'pca_model' in pca_data:
                return pca_data['pca_model']
            else:
                # If it's directly a PCA model
                return pca_data
    except Exception as e:
        raise ValueError(f"Error loading PCA model: {e}")

# Load projection matrix and hyperplanes from .dat
def load_projection_and_hyperplanes(dat_path):
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Data file not found: {dat_path}")
    
    with open(dat_path, 'rb') as f:
        raw = f.read()
    
    print(f"File size: {len(raw)} bytes")
    
    header_size = 128
    matrix_size = 128 * 96 * 4  # float32 = 4 bytes per element
    expected_total_size = header_size + 2 * matrix_size  # header + proj_matrix + hyperplanes
    
    print(f"Expected file size: {expected_total_size} bytes")
    print(f"Matrix size: {matrix_size} bytes each")
    
    if len(raw) < header_size + matrix_size:
        raise ValueError(f"File too small. Expected at least {header_size + matrix_size} bytes, got {len(raw)}")
    
    # Extract projection matrix
    proj_start = header_size
    proj_end = header_size + matrix_size
    proj_flat = np.frombuffer(raw[proj_start:proj_end], dtype=np.float32)
    
    if len(proj_flat) != 128 * 96:
        raise ValueError(f"Projection matrix has wrong size: {len(proj_flat)}, expected {128 * 96}")
    
    proj_matrix = proj_flat.reshape(128, 96)
    
    # Check if hyperplanes data exists
    if len(raw) >= header_size + 2 * matrix_size:
        # Extract hyperplanes
        hyper_start = header_size + matrix_size
        hyper_end = header_size + 2 * matrix_size
        hyper_flat = np.frombuffer(raw[hyper_start:hyper_end], dtype=np.float32)
        
        if len(hyper_flat) != 128 * 96:
            print(f"Warning: Hyperplanes data has size {len(hyper_flat)}, expected {128 * 96}")
            print("Generating random hyperplanes as fallback...")
            hyperplanes = generate_random_hyperplanes()
        else:
            hyperplanes = hyper_flat.reshape(128, 96)
    else:
        print("Warning: File doesn't contain hyperplanes data. Generating random hyperplanes...")
        hyperplanes = generate_random_hyperplanes()
    
    return proj_matrix, hyperplanes

# Generate random hyperplanes if not available in file
def generate_random_hyperplanes():
    """Generate random hyperplanes for hashing when not available in data file"""
    np.random.seed(42)  # For reproducible results
    hyperplanes = np.random.randn(128, 96)
    # Normalize each hyperplane
    for i in range(128):
        hyperplanes[i] = hyperplanes[i] / np.linalg.norm(hyperplanes[i])
    return hyperplanes

# Generate 128-bit hash from 96D vector using hyperplanes
def compute_hash(embedding_96, hyperplanes):
    embedding_96 = embedding_96 / np.linalg.norm(embedding_96)
    bits = (np.dot(hyperplanes, embedding_96) > 0).astype(np.uint8)
    return bits

def bits_to_hex(bits):
    return ''.join([f"{int(''.join(map(str, bits[i:i+8])), 2):02x}" for i in range(0, 128, 8)])

def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2)

def cosine_sim(emb1, emb2):
    return cosine_similarity([emb1], [emb2])[0][0]

def process_image(img_path, pca, proj_matrix, hyperplanes):
    try:
        face = load_face(img_path)
        emb_512 = get_embedding(face)
        emb_128 = pca.transform([emb_512])[0]
        emb_96 = np.dot(emb_128, proj_matrix)
        hash_bits = compute_hash(emb_96, hyperplanes)
        hash_hex = bits_to_hex(hash_bits)
        return {
            'success': True,
            'embedding_512': emb_512,
            'embedding_128': emb_128,
            'embedding_96': emb_96,
            'hash_bits': hash_bits,
            'hash_hex': hash_hex
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}

def compare_images(img1, img2, pca, proj_matrix, hyperplanes):
    result1 = process_image(img1, pca, proj_matrix, hyperplanes)
    result2 = process_image(img2, pca, proj_matrix, hyperplanes)

    if not result1['success'] or not result2['success']:
        print("Error processing images:")
        if not result1['success']:
            print(f"Image 1 error: {result1.get('error')}")
        if not result2['success']:
            print(f"Image 2 error: {result2.get('error')}")
        return

    print("96-Dimensional Neural Hash:")
    print(f"Image 1: {result1['embedding_96']}")
    print(f"Image 2: {result2['embedding_96']}")

# Main
if __name__ == "__main__":
    image1_path = "./data/probe/image11.jpg"
    image2_path = "./data/probe/image12.jpg"
    dat_file = "./models/neuralhash_128x96_seed1.dat"
    pca_file = "./models/pca_512_to_128.pkl"

    try:
        pca_model = load_pca(pca_file)
        proj_matrix, hyperplanes = load_projection_and_hyperplanes(dat_file)
        compare_images(image1_path, image2_path, pca_model, proj_matrix, hyperplanes)
        
    except Exception as e:
        print(f"Error: {e}")

def get_96d_neural_hash(img_path, pca, proj_matrix):
    """
    Extract just the 96-dimensional neural hash embedding from an image
    """
    try:
        face = load_face(img_path)
        emb_512 = get_embedding(face)
        emb_128 = pca.transform([emb_512])[0]
        emb_96 = np.dot(emb_128, proj_matrix)  # This is your 96D neural hash
        return emb_96
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None