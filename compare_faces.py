import torch
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# 1. Load MTCNN (for face detection + alignment)
mtcnn = MTCNN(image_size=160, margin=0)

# 2. Load ArcFace model (InceptionResnetV1 trained on VGGFace2)
model = InceptionResnetV1(pretrained='vggface2').eval()

def get_face_embedding(image_path):
    """Detect, align, and extract ArcFace embedding for a single image."""
    img = Image.open(image_path).convert('RGB')

    # Detect and align face
    face_tensor = mtcnn(img)
    if face_tensor is None:
        raise ValueError(f"No face detected in {image_path}")

    # Get embedding
    with torch.no_grad():
        embedding = model(face_tensor.unsqueeze(0))  # shape: (1, 512)
        embedding = embedding[0] / embedding[0].norm()  # L2-normalize

    return embedding.numpy()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

if __name__ == "__main__":
    # Replace these with your actual images
    image1 = "data/probe/image4.jpg"
    image2 = "data/probe/image5.jpg"

    try:
        emb1 = get_face_embedding(image1)
        emb2 = get_face_embedding(image2)

        sim = cosine_similarity(emb1, emb2)
        print(f"\nCosine Similarity: {sim:.4f}")

        threshold = 0.3  # You can tune this
        if sim > threshold:
            print("✅ Likely the same person")
        else:
            print("❌ Likely different people")

    except Exception as e:
        print("Error:", e)
