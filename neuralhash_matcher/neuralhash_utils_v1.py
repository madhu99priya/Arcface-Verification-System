from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import pickle
from PIL import Image

# Initialize MTCNN to keep all detected faces
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True) 
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def load_pca_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)['pca_model']


def load_hyperplanes(path):
    arr = np.fromfile(path, dtype=np.int8).reshape(-1, 128).astype(np.float32)[:96]
    return arr


def generate_neuralhash(image_path, pca, hyperplanes):
    img = Image.open(image_path).convert('RGB')
    
    # Detect all faces and get their bounding boxes
    boxes, _ = mtcnn.detect(img)
    
    # mtcnn() returns a list of tensors when keep_all=True
    face_tensors = mtcnn(img)

    if face_tensors is None:
        raise ValueError('No face detected')

    # If multiple faces are detected, select the one with the largest bounding box
    if len(face_tensors) > 1:
        # Calculate the area of each bounding box
        box_areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in boxes]
        
        # Find the index of the largest box
        largest_box_index = np.argmax(box_areas)
        
        # Select the corresponding face tensor
        face = face_tensors[largest_box_index]
    else:
        # If only one face is detected
        face = face_tensors[0]

    emb_512 = resnet(face.unsqueeze(0)).detach().cpu().numpy().squeeze()
    emb_128 = pca.transform([emb_512])[0]
    emb_128 /= np.linalg.norm(emb_128)
    bits = (np.dot(hyperplanes, emb_128) > 0).astype(np.uint8)
    return bits


def bits_to_hex(bits):
    return ''.join(f"{int(''.join(map(str, bits[i:i+8])),2):02x}" for i in range(0,len(bits),8))