# neuralhash_utils.py
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle
from PIL import Image

mtcnn = MTCNN(image_size=160, margin=0)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def load_pca_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['pca_model']


def load_hyperplanes(path):
    arr = np.fromfile(path, dtype=np.int8).reshape(-1, 128).astype(np.float32)[:96]
    return arr


def generate_neuralhash(image_path, pca, hyperplanes):
    img = Image.open(image_path).convert('RGB')
    face = mtcnn(img)
    if face is None:
        raise ValueError('No face detected')
    emb_512 = resnet(face.unsqueeze(0)).detach().cpu().numpy().squeeze()
    emb_128 = pca.transform([emb_512])[0]
    emb_128 /= np.linalg.norm(emb_128)
    bits = (np.dot(hyperplanes, emb_128) > 0).astype(np.uint8)
    return bits


def bits_to_hex(bits):
    return ''.join(f"{int(''.join(map(str, bits[i:i+8])),2):02x}" for i in range(0,len(bits),8))