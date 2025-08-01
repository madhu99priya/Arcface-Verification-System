import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import os
import pickle

# ==== Load Models ====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Face detector (MTCNN)
mtcnn = MTCNN(image_size=160, margin=0, device=device)

# ArcFace model (InceptionResnetV1)
arcface = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==== Step 1: Load and Align Face ====
def load_and_align_image(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    aligned_face = mtcnn(img_rgb)
    if aligned_face is None:
        raise ValueError(f"No face detected in: {img_path}")
    return aligned_face.unsqueeze(0).to(device)

# ==== Step 2: Get ArcFace Embedding ====
def get_embedding(aligned_face_tensor):
    with torch.no_grad():
        embedding = arcface(aligned_face_tensor)
    return embedding.cpu().numpy().squeeze()  # shape: (512,)

# ==== Step 3: Load PCA (or train if first time) ====
def fit_pca_on_dataset(embeddings_np):
    pca = PCA(n_components=128)
    pca.fit(embeddings_np)
    return pca

# ==== Step 4: Load NeuralHash Hyperplanes ====
def load_hyperplanes(dat_path):
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Hyperplanes file not found: {dat_path}")
    
    with open(dat_path, "rb") as f:
        data = f.read()

    hyperplanes = np.frombuffer(data, dtype=np.float32).reshape(128, 128)
    return hyperplanes

# ==== Step 5: Compute NeuralHash ====
def compute_neuralhash(embedding_128, hyperplanes):
    embedding_128 = embedding_128 / np.linalg.norm(embedding_128)
    bits = (np.dot(hyperplanes, embedding_128) > 0).astype(np.uint8)
    return bits

def bits_to_hex(bits):
    return ''.join([f"{int(''.join(map(str, bits[i:i+8])), 2):02x}" for i in range(0, 128, 8)])

# ==== Similarity Metrics ====
def hamming_distance(hash1, hash2):
    return np.sum(hash1 != hash2)

def hamming_similarity(hash1, hash2):
    return 1 - (hamming_distance(hash1, hash2) / len(hash1))

def euclidean_distance(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)

def cosine_similarity_score(emb1, emb2):
    return cosine_similarity([emb1], [emb2])[0][0]

def manhattan_distance(emb1, emb2):
    return np.sum(np.abs(emb1 - emb2))

# ==== Complete Processing Pipeline ====
def process_image(img_path, pca, hyperplanes):
    try:
        face_tensor = load_and_align_image(img_path)
        emb_512 = get_embedding(face_tensor)
        emb_128 = pca.transform([emb_512])[0]
        hash_bits = compute_neuralhash(emb_128, hyperplanes)
        neural_hash = bits_to_hex(hash_bits)
        
        return {
            'success': True,
            'embedding_512': emb_512,
            'embedding_128': emb_128,
            'hash_bits': hash_bits,
            'neural_hash': neural_hash,
            'image_path': img_path
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'image_path': img_path
        }

def load_trained_pca(pca_path='./models/pca_512_to_128.pkl'):
    with open(pca_path, 'rb') as f:
        pca_data = pickle.load(f)
        return pca_data['pca_model']

def compare_images(img1_path, img2_path, dat_file):
    print(f"Comparing images:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    print("-" * 60)
    
    pca = load_trained_pca()
    hyperplanes = load_hyperplanes(dat_file)
        
    result1 = process_image(img1_path, pca, hyperplanes)
    result2 = process_image(img2_path, pca, hyperplanes)
    
    if not result1['success']:
        print(f"Error processing image 1: {result1['error']}")
        return
    if not result2['success']:
        print(f"Error processing image 2: {result2['error']}")
        return
    
    print("Neural Hashes:")
    print(f"  Image 1: {result1['neural_hash']}")
    print(f"  Image 2: {result2['neural_hash']}")
    print()
    
    print("Similarity Metrics:")
    print("-" * 30)
    
    hamming_dist = hamming_distance(result1['hash_bits'], result2['hash_bits'])
    hamming_sim = hamming_similarity(result1['hash_bits'], result2['hash_bits'])
    print(f"Hamming Distance:     {hamming_dist}/128 bits")
    print(f"Hamming Similarity:   {hamming_sim:.4f} (1.0 = identical)")
    
    cos_sim_512 = cosine_similarity_score(result1['embedding_512'], result2['embedding_512'])
    print(f"Cosine Similarity (512-d): {cos_sim_512:.4f}")
    
    cos_sim_128 = cosine_similarity_score(result1['embedding_128'], result2['embedding_128'])
    print(f"Cosine Similarity (128-d): {cos_sim_128:.4f}")
    
    eucl_dist_512 = euclidean_distance(result1['embedding_512'], result2['embedding_512'])
    eucl_dist_128 = euclidean_distance(result1['embedding_128'], result2['embedding_128'])
    print(f"Euclidean Distance (512-d): {eucl_dist_512:.4f}")
    print(f"Euclidean Distance (128-d): {eucl_dist_128:.4f}")
    
    manhattan_dist_512 = manhattan_distance(result1['embedding_512'], result2['embedding_512'])
    manhattan_dist_128 = manhattan_distance(result1['embedding_128'], result2['embedding_128'])
    print(f"Manhattan Distance (512-d): {manhattan_dist_512:.4f}")
    print(f"Manhattan Distance (128-d): {manhattan_dist_128:.4f}")
    
    print()
    print("Interpretation:")
    print("-" * 15)
    
    if hamming_sim > 0.85:
        hash_verdict = "Very Similar"
    elif hamming_sim > 0.75:
        hash_verdict = "Similar"
    elif hamming_sim > 0.65:
        hash_verdict = "Moderately Similar"
    else:
        hash_verdict = "Different"
    
    if cos_sim_512 > 0.8:
        embedding_verdict = "Very Similar"
    elif cos_sim_512 > 0.6:
        embedding_verdict = "Similar"
    elif cos_sim_512 > 0.4:
        embedding_verdict = "Moderately Similar"
    else:
        embedding_verdict = "Different"
    
    print(f"Hash-based verdict:      {hash_verdict}")
    print(f"Embedding-based verdict: {embedding_verdict}")
    
    return {
        'hamming_distance': hamming_dist,
        'hamming_similarity': hamming_sim,
        'cosine_similarity_512': cos_sim_512,
        'cosine_similarity_128': cos_sim_128,
        'euclidean_distance_512': eucl_dist_512,
        'euclidean_distance_128': eucl_dist_128,
        'manhattan_distance_512': manhattan_dist_512,
        'manhattan_distance_128': manhattan_dist_128,
        'hash_verdict': hash_verdict,
        'embedding_verdict': embedding_verdict
    }

if __name__ == "__main__":
    image1_path = "./data/probe/image11.jpg"
    image2_path = "./data/probe/image12.jpg"
    dat_file = "./models/neuralhash_128x128_seed1.dat"
    
    try:
        results = compare_images(image1_path, image2_path, dat_file)
        print(f"\nComparison completed successfully!")
    except Exception as e:
        print(f"Error during comparison: {e}")
        
    print("\n" + "="*60)
    print("Individual Image Processing:")
    print("="*60)
    
    try:
        pca = load_trained_pca()
        hyperplanes = load_hyperplanes(dat_file)
        
        result1 = process_image(image1_path, pca, hyperplanes)
        if result1['success']:
            print(f"Image 1 hash: {result1['neural_hash']}")
        else:
            print(f"Image 1 error: {result1['error']}")
            
        result2 = process_image(image2_path, pca, hyperplanes)
        if result2['success']:
            print(f"Image 2 hash: {result2['neural_hash']}")
        else:
            print(f"Image 2 error: {result2['error']}")
            
    except Exception as e:
        print(f"Error in individual processing: {e}")
