import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from sklearn.decomposition import PCA
import os
import pickle
import glob
from tqdm import tqdm
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

print("Loading face detection and recognition models...")
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
arcface = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
print("Models loaded successfully!")

def load_and_align_image(img_path):
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    aligned_face = mtcnn(img_rgb)
    if aligned_face is None:
        raise ValueError(f"No face detected in: {img_path}")
    
    return aligned_face.unsqueeze(0).to(DEVICE)

def get_face_embedding(aligned_face_tensor):
    with torch.no_grad():
        embedding = arcface(aligned_face_tensor)
    return embedding.cpu().numpy().squeeze()

def collect_face_embeddings(dataset_folder, max_images=None):
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_folder, '**', ext), recursive=True))

    if max_images and len(image_paths) > max_images:
        image_paths = image_paths[:max_images]

    embeddings = []
    failed_images = []

    with tqdm(total=len(image_paths), desc="Extracting face embeddings") as pbar:
        for img_path in image_paths:
            try:
                face_tensor = load_and_align_image(img_path)
                embedding = get_face_embedding(face_tensor)
                embeddings.append(embedding)
            except Exception as e:
                failed_images.append((img_path, str(e)))
            pbar.update(1)

    embeddings_array = np.array(embeddings)
    return embeddings_array, failed_images

def train_pca_model(embeddings_array, n_components=128, whiten=False):
    print(f"Training PCA: {embeddings_array.shape[1]}D → {n_components}D")
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(embeddings_array)
    return pca

def save_pca_model(pca, embeddings_array, save_path, metadata=None):
    save_data = {
        'pca_model': pca,
        'n_training_samples': len(embeddings_array),
        'n_components': pca.n_components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_explained_variance': np.sum(pca.explained_variance_ratio_),
        'training_data_shape': embeddings_array.shape,
        'metadata': metadata or {}
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"PCA model saved to: {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='Path to dataset of face images')
    parser.add_argument('--output', default='./models/pca_512_to_128.pkl', help='Where to save the trained PCA')
    parser.add_argument('--max_images', type=int, default=None)
    args = parser.parse_args()

    embeddings_array, failed_images = collect_face_embeddings(args.dataset, args.max_images)
    pca = train_pca_model(embeddings_array, n_components=128)
    save_pca_model(pca, embeddings_array, args.output)

if __name__ == "__main__":
    main()
