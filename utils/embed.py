##j ust like align.py handles preprocessing, the utils/embed.py file in a face recognition system is typically responsible for generating face embeddings using a pretrained model like ArcFace

import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np

# Load pretrained ArcFace from facenet-pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def get_embedding(img_bgr):
    # Input: aligned BGR numpy array
    img = torch.tensor(img_bgr, device=device).permute(2, 0, 1).float()
    img = img.unsqueeze(0) / 255.0
    with torch.no_grad():
        emb = cnn(img)
    return emb.cpu().numpy().squeeze()  # shape (512,)