# # main.py

# from facenet_pytorch import MTCNN, InceptionResnetV1
# import torch
# import numpy as np
# from sklearn.decomposition import PCA
# import pickle
# from PIL import Image
# import os

# # Load models
# mtcnn = MTCNN(image_size=160, margin=0)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()

# def load_pca_model(path):
#     with open(path, 'rb') as f:
#         return pickle.load(f)['pca_model']

# def load_hyperplanes(path):
#     return np.fromfile(path, dtype=np.int8).reshape(-1, 128).astype(np.float32)[:96]

# def generate_neuralhash(image_path, pca, hyperplanes):
#     img = Image.open(image_path).convert('RGB')
#     face = mtcnn(img)
#     if face is None:
#         raise ValueError("No face detected")

#     emb_512 = resnet(face.unsqueeze(0)).detach().numpy().squeeze()
#     emb_128 = pca.transform([emb_512])[0]
#     emb_128 /= np.linalg.norm(emb_128)
#     bits = (np.dot(hyperplanes, emb_128) > 0).astype(np.uint8)
#     return bits

# def bits_to_hex(bits):
#     return ''.join(f'{int("".join(map(str, bits[i:i+8])), 2):02x}' for i in range(0, len(bits), 8))

# def hamming_distance(bits1, bits2):
#     return sum(b1 != b2 for b1, b2 in zip(bits1, bits2))

# def compare_images(img1_path, img2_path, pca_path, hyperplane_path):
#     if not os.path.exists(pca_path) or not os.path.exists(hyperplane_path):
#         raise FileNotFoundError("PCA or hyperplane file not found")

#     pca = load_pca_model(pca_path)
#     hyperplanes = load_hyperplanes(hyperplane_path)

#     bits1 = generate_neuralhash(img1_path, pca, hyperplanes)
#     bits2 = generate_neuralhash(img2_path, pca, hyperplanes)

#     hash1 = bits_to_hex(bits1)
#     hash2 = bits_to_hex(bits2)
#     hamming = hamming_distance(bits1, bits2)

#     return hash1, hash2, hamming
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import numpy as np
from sklearn.decomposition import PCA
import pickle
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def load_pca_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['pca_model']

def load_hyperplanes(path):
    return np.fromfile(path, dtype=np.int8).reshape(-1, 128).astype(np.float32)[:96]

def generate_neuralhash(image_path, pca, hyperplanes):
    img = Image.open(image_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        raise ValueError("No face detected")
    emb_512 = resnet(face.unsqueeze(0)).detach().numpy().squeeze()
    emb_128 = pca.transform([emb_512])[0]
    emb_128 /= np.linalg.norm(emb_128)
    bits = (np.dot(hyperplanes, emb_128) > 0).astype(np.uint8)
    return bits

def bits_to_hex(bits):
    return ''.join(f'{int("".join(map(str, bits[i:i+8])), 2):02x}' for i in range(0, len(bits), 8))
