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

# ==== Configuration ====
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# ==== Load Models ====
print("Loading face detection and recognition models...")
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
arcface = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
print("Models loaded successfully!")

def load_and_align_image(img_path):
    """Load image and extract aligned face"""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise ValueError(f"Could not load image: {img_path}")
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    aligned_face = mtcnn(img_rgb)
    if aligned_face is None:
        raise ValueError(f"No face detected in: {img_path}")
    
    return aligned_face.unsqueeze(0).to(DEVICE)

def get_face_embedding(aligned_face_tensor):
    """Extract 512-dimensional face embedding using ArcFace"""
    with torch.no_grad():
        embedding = arcface(aligned_face_tensor)
    return embedding.cpu().numpy().squeeze()

def collect_face_embeddings(dataset_folder, max_images=None, batch_size=32):
    """
    Collect face embeddings from all images in dataset folder
    
    Args:
        dataset_folder: Path to folder containing face images
        max_images: Maximum number of images to process (None for all)
        batch_size: Process images in batches to save memory
    
    Returns:
        numpy array of embeddings, shape (n_faces, 512)
    """
    print(f"\nCollecting face embeddings from: {dataset_folder}")
    
    # Find all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.JPG', '*.JPEG', '*.PNG']
    image_paths = []
    
    print("Scanning for image files...")
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(dataset_folder, '**', ext), recursive=True))
    
    if max_images and len(image_paths) > max_images:
        print(f"Limiting to {max_images} images out of {len(image_paths)} found")
        image_paths = image_paths[:max_images]
    
    print(f"Found {len(image_paths)} images to process")
    
    if len(image_paths) == 0:
        raise ValueError(f"No images found in {dataset_folder}")
    
    embeddings = []
    failed_images = []
    processed_count = 0
    
    # Process images with progress bar
    with tqdm(total=len(image_paths), desc="Extracting face embeddings") as pbar:
        for img_path in image_paths:
            try:
                # Load and align face
                face_tensor = load_and_align_image(img_path)
                
                # Get embedding
                embedding = get_face_embedding(face_tensor)
                embeddings.append(embedding)
                processed_count += 1
                
            except Exception as e:
                failed_images.append((img_path, str(e)))
                
            pbar.update(1)
            
            # Optional: Clear GPU cache periodically
            if processed_count % batch_size == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Report results
    print(f"\nProcessing Summary:")
    print(f"  Successfully processed: {len(embeddings)} images")
    print(f"  Failed to process: {len(failed_images)} images")
    
    if failed_images:
        print(f"\nFirst 5 failed images:")
        for i, (img_path, error) in enumerate(failed_images[:5]):
            print(f"  {i+1}. {os.path.basename(img_path)}: {error}")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more failures")
    
    if len(embeddings) == 0:
        raise ValueError("No face embeddings could be extracted!")
    
    embeddings_array = np.array(embeddings)
    print(f"\nCollected embeddings shape: {embeddings_array.shape}")
    
    return embeddings_array, failed_images

def train_pca_model(embeddings_array, n_components=96, whiten=False):
    """
    Train PCA model on face embeddings
    
    Args:
        embeddings_array: numpy array of face embeddings, shape (n_faces, 512)
        n_components: number of PCA components to keep
        whiten: whether to whiten the components
    
    Returns:
        trained PCA model
    """
    print(f"\nTraining PCA model:")
    print(f"  Input embeddings: {embeddings_array.shape}")
    print(f"  PCA components: {n_components}")
    print(f"  Whitening: {whiten}")
    
    if len(embeddings_array) < n_components:
        print(f"Warning: Only {len(embeddings_array)} embeddings available for {n_components} components")
        n_components = len(embeddings_array) - 1
        print(f"Reducing n_components to {n_components}")
    
    # Train PCA
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(embeddings_array)
    
    # Calculate statistics
    total_variance = np.sum(pca.explained_variance_ratio_)
    
    print(f"\nPCA Training Results:")
    print(f"  Total explained variance: {total_variance:.4f} ({total_variance*100:.2f}%)")
    print(f"  Mean explained variance per component: {np.mean(pca.explained_variance_ratio_):.4f}")
    print(f"  Top 10 component variances: {pca.explained_variance_ratio_[:10].round(4)}")
    
    # Check if we're capturing enough variance
    if total_variance < 0.8:
        print(f"Warning: PCA only captures {total_variance*100:.1f}% of variance. Consider increasing n_components.")
    
    return pca

def save_pca_model(pca, embeddings_array, save_path, metadata=None):
    """
    Save PCA model with metadata
    
    Args:
        pca: trained PCA model
        embeddings_array: original embeddings used for training
        save_path: path to save the model
        metadata: additional metadata to save
    """
    # Prepare data to save
    save_data = {
        'pca_model': pca,
        'n_training_samples': len(embeddings_array),
        'n_components': pca.n_components_,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'total_explained_variance': np.sum(pca.explained_variance_ratio_),
        'training_data_shape': embeddings_array.shape,
        'metadata': metadata or {}
    }
    
    # Save to file
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\nPCA model saved to: {save_path}")
    print(f"Model file size: {os.path.getsize(save_path) / (1024*1024):.2f} MB")

def load_and_test_pca(save_path):
    """Load and display PCA model information"""
    with open(save_path, 'rb') as f:
        save_data = pickle.load(f)
    
    pca = save_data['pca_model']
    
    print(f"\nLoaded PCA Model Information:")
    print(f"  Training samples: {save_data['n_training_samples']}")
    print(f"  PCA components: {save_data['n_components']}")
    print(f"  Total explained variance: {save_data['total_explained_variance']:.4f}")
    print(f"  Training data shape: {save_data['training_data_shape']}")
    
    return pca, save_data

def main():
    parser = argparse.ArgumentParser(description='Train PCA model on face embeddings')
    parser.add_argument('--dataset', '-d', required=True, help='Path to dataset folder containing face images')
    parser.add_argument('--output', '-o', default='./models/trained_pca_96.pkl', help='Output path for trained PCA model')
    parser.add_argument('--components', '-c', type=int, default=96, help='Number of PCA components (default: 96)')
    parser.add_argument('--max_images', '-m', type=int, default=None, help='Maximum number of images to process')
    parser.add_argument('--whiten', action='store_true', help='Apply whitening to PCA components')
    parser.add_argument('--test', action='store_true', help='Test the saved model after training')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.dataset):
        print(f"Error: Dataset folder not found: {args.dataset}")
        return
    
    print("="*80)
    print("PCA TRAINING FOR FACE EMBEDDINGS")
    print("="*80)
    print(f"Dataset folder: {args.dataset}")
    print(f"Output file: {args.output}")
    print(f"PCA components: {args.components}")
    print(f"Max images: {args.max_images or 'All'}")
    print(f"Whitening: {args.whiten}")
    
    try:
        # Step 1: Collect embeddings
        embeddings_array, failed_images = collect_face_embeddings(
            args.dataset, 
            max_images=args.max_images
        )
        
        # Step 2: Train PCA
        pca = train_pca_model(
            embeddings_array, 
            n_components=args.components,
            whiten=args.whiten
        )
        
        # Step 3: Save model
        metadata = {
            'dataset_folder': args.dataset,
            'failed_images_count': len(failed_images),
            'training_args': vars(args)
        }
        
        save_pca_model(pca, embeddings_array, args.output, metadata)
        
        # Step 4: Test model (optional)
        if args.test:
            print("\n" + "="*50)
            print("TESTING SAVED MODEL")
            print("="*50)
            loaded_pca, model_info = load_and_test_pca(args.output)
            
            # Test with a sample embedding
            if len(embeddings_array) > 0:
                test_embedding = embeddings_array[0:1]  # First embedding
                reduced_embedding = loaded_pca.transform(test_embedding)
                print(f"Test transformation: {test_embedding.shape} -> {reduced_embedding.shape}")
                print("Model loaded and tested successfully!")
        
        print("\n" + "="*80)
        print("PCA TRAINING COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"Trained on {len(embeddings_array)} face embeddings")
        print(f"Model saved to: {args.output}")
        print("You can now use this PCA model for consistent face hashing!")
        
    except Exception as e:
        print(f"\nError during training: {e}")
        return

if __name__ == "__main__":
    # If no command line arguments, use interactive mode
    import sys
    if len(sys.argv) == 1:
        print("="*60)
        print("PCA TRAINING SCRIPT - INTERACTIVE MODE")
        print("="*60)
        
        # Get inputs interactively
        dataset_folder = input("Enter path to dataset folder: ").strip()
        if not os.path.exists(dataset_folder):
            print(f"Error: Folder not found: {dataset_folder}")
            exit(1)
        
        output_path = input("Enter output path [./models/trained_pca_96.pkl]: ").strip()
        if not output_path:
            output_path = "./models/trained_pca_96.pkl"
        
        n_components = input("Enter number of PCA components [96]: ").strip()
        n_components = int(n_components) if n_components else 96
        
        max_images = input("Enter max images to process [all]: ").strip()
        max_images = int(max_images) if max_images else None
        
        # Set up arguments
        class Args:
            dataset = dataset_folder
            output = output_path
            components = n_components
            max_images = max_images
            whiten = False
            test = True
        
        args = Args()
        
        # Run main with interactive args
        sys.argv = ['train_pca.py']  # Reset argv to avoid argparse conflicts
        
        print("="*60)
        print("STARTING PCA TRAINING...")
        print("="*60)
        
        try:
            # Collect embeddings
            embeddings_array, failed_images = collect_face_embeddings(
                args.dataset, 
                max_images=args.max_images
            )
            
            # Train PCA
            pca = train_pca_model(embeddings_array, n_components=args.components)
            
            # Save model
            metadata = {
                'dataset_folder': args.dataset,
                'failed_images_count': len(failed_images)
            }
            save_pca_model(pca, embeddings_array, args.output, metadata)
            
            # Test model
            if args.test:
                print("\nTesting saved model...")
                loaded_pca, model_info = load_and_test_pca(args.output)
                print("Model test successful!")
            
            print("\n" + "="*60)
            print("TRAINING COMPLETED SUCCESSFULLY!")
            print("="*60)
            
        except Exception as e:
            print(f"Error: {e}")
    else:
        # Command line mode
        main()