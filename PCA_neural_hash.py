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
    pca = PCA(n_components=96)
    pca.fit(embeddings_np)
    return pca

def simulate_pca():
    dummy_data = np.random.randn(1000, 512)
    return fit_pca_on_dataset(dummy_data)

# ==== Step 4: Load NeuralHash Hyperplanes ====
def load_hyperplanes(dat_path):
    if not os.path.exists(dat_path):
        raise FileNotFoundError(f"Hyperplanes file not found: {dat_path}")
    
    with open(dat_path, "rb") as f:
        data = f.read()
    
    # Skip header (e.g., 128 bytes)
    header_size = 128  
    raw = data[header_size:]  # skip header if it's present
    
    hyperplanes = np.frombuffer(raw, dtype=np.float32).reshape(128, 96)
    return hyperplanes

# ==== Step 5: Compute NeuralHash ====
def compute_neuralhash(embedding_96, hyperplanes):
    embedding_96 = embedding_96 / np.linalg.norm(embedding_96)
    bits = (np.dot(hyperplanes, embedding_96) > 0).astype(np.uint8)
    return bits

def bits_to_hex(bits):
    return ''.join([f"{int(''.join(map(str, bits[i:i+8])), 2):02x}" for i in range(0, 128, 8)])

# ==== Similarity Metrics ====
def hamming_distance(hash1, hash2):
    """Calculate Hamming distance between two binary hash arrays"""
    return np.sum(hash1 != hash2)

def hamming_similarity(hash1, hash2):
    """Calculate Hamming similarity (0-1, where 1 is identical)"""
    return 1 - (hamming_distance(hash1, hash2) / len(hash1))

def euclidean_distance(emb1, emb2):
    """Calculate Euclidean distance between embeddings"""
    return np.linalg.norm(emb1 - emb2)

def cosine_similarity_score(emb1, emb2):
    """Calculate cosine similarity between embeddings"""
    return cosine_similarity([emb1], [emb2])[0][0]

def manhattan_distance(emb1, emb2):
    """Calculate Manhattan (L1) distance between embeddings"""
    return np.sum(np.abs(emb1 - emb2))

# ==== Complete Processing Pipeline ====
def process_image(img_path, pca, hyperplanes):
    """Process a single image and return all relevant data"""
    try:
        # 1. Load and align face
        face_tensor = load_and_align_image(img_path)
        
        # 2. Get ArcFace embedding (512-d)
        emb_512 = get_embedding(face_tensor)
        
        # 3. Reduce to 96-d with PCA
        emb_96 = pca.transform([emb_512])[0]
        
        # 4. Compute hash
        hash_bits = compute_neuralhash(emb_96, hyperplanes)
        neural_hash = bits_to_hex(hash_bits)
        
        return {
            'success': True,
            'embedding_512': emb_512,
            'embedding_96': emb_96,
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

def load_trained_pca(pca_path='./models/trained_pca_96.pkl'):
    """Load the trained PCA model from pickle file"""
    with open(pca_path, 'rb') as f:
        pca_data = pickle.load(f)
        return pca_data['pca_model']


def compare_images(img1_path, img2_path, dat_file):
    """Compare two images using multiple similarity metrics"""
    print(f"Comparing images:")
    print(f"  Image 1: {img1_path}")
    print(f"  Image 2: {img2_path}")
    print("-" * 60)
    
    # Load PCA and hyperplanes
    pca = load_trained_pca()  
    hyperplanes = load_hyperplanes(dat_file)
        
    # Process both images
    result1 = process_image(img1_path, pca, hyperplanes)
    result2 = process_image(img2_path, pca, hyperplanes)
    
    # Check if both images were processed successfully
    if not result1['success']:
        print(f"Error processing image 1: {result1['error']}")
        return
    
    if not result2['success']:
        print(f"Error processing image 2: {result2['error']}")
        return
    
    # Display individual hashes
    print("Neural Hashes:")
    print(f"  Image 1: {result1['neural_hash']}")
    print(f"  Image 2: {result2['neural_hash']}")
    print()
    
    # Calculate all similarity metrics
    print("Similarity Metrics:")
    print("-" * 30)
    
    # 1. Hamming distance/similarity on hash bits
    hamming_dist = hamming_distance(result1['hash_bits'], result2['hash_bits'])
    hamming_sim = hamming_similarity(result1['hash_bits'], result2['hash_bits'])
    print(f"Hamming Distance:     {hamming_dist}/128 bits")
    print(f"Hamming Similarity:   {hamming_sim:.4f} (1.0 = identical)")
    
    # 2. Cosine similarity on 512-d embeddings
    cos_sim_512 = cosine_similarity_score(result1['embedding_512'], result2['embedding_512'])
    print(f"Cosine Similarity (512-d): {cos_sim_512:.4f} (1.0 = identical)")
    
    # 3. Cosine similarity on 96-d embeddings
    cos_sim_96 = cosine_similarity_score(result1['embedding_96'], result2['embedding_96'])
    print(f"Cosine Similarity (96-d):  {cos_sim_96:.4f} (1.0 = identical)")
    
    # 4. Euclidean distance on embeddings
    eucl_dist_512 = euclidean_distance(result1['embedding_512'], result2['embedding_512'])
    eucl_dist_96 = euclidean_distance(result1['embedding_96'], result2['embedding_96'])
    print(f"Euclidean Distance (512-d): {eucl_dist_512:.4f} (0.0 = identical)")
    print(f"Euclidean Distance (96-d):  {eucl_dist_96:.4f} (0.0 = identical)")
    
    # 5. Manhattan distance on embeddings
    manhattan_dist_512 = manhattan_distance(result1['embedding_512'], result2['embedding_512'])
    manhattan_dist_96 = manhattan_distance(result1['embedding_96'], result2['embedding_96'])
    print(f"Manhattan Distance (512-d): {manhattan_dist_512:.4f} (0.0 = identical)")
    print(f"Manhattan Distance (96-d):  {manhattan_dist_96:.4f} (0.0 = identical)")
    
    print()
    print("Interpretation:")
    print("-" * 15)
    
    # Provide interpretation based on typical thresholds
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
        'cosine_similarity_96': cos_sim_96,
        'euclidean_distance_512': eucl_dist_512,
        'euclidean_distance_96': eucl_dist_96,
        'manhattan_distance_512': manhattan_dist_512,
        'manhattan_distance_96': manhattan_dist_96,
        'hash_verdict': hash_verdict,
        'embedding_verdict': embedding_verdict
    }

if __name__ == "__main__":
    # Paths to the two images you want to compare
    image1_path = "./data/probe/image4.jpg"
    image2_path = "./data/probe/image5.jpg"
    
    # Path to NeuralHash dat file -> getting hyperplanes
    dat_file = "./models/neuralhash_128x96_seed1.dat"
    
    # Compare the two images
    try:
        results = compare_images(image1_path, image2_path, dat_file)
        print(f"\nComparison completed successfully!")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        
    # Optional: You can also process individual images
    print("\n" + "="*60)
    print("Individual Image Processing:")
    print("="*60)
    
    try:
        pca = load_trained_pca()  # Use same helper function
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