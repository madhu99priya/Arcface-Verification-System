from utils.align import align_face
from utils.embed import get_embedding
from utils.hash import compute_neural_hash
from utils.index import BinaryIndex
import faiss, numpy as np
import os

# Load index & ids
idx = faiss.read_index('models/gallery.index')
ids = open('models/gallery.ids').read().splitlines()
bx = BinaryIndex(); bx.index = idx; bx.ids = ids

# Process probe images
for img in os.listdir('data/probe'):
    aligned = align_face(os.path.join('data/probe', img))
    if aligned is None:
        print(f"No face: {img}")
        continue
    emb = get_embedding(aligned)
    bits = compute_neural_hash(emb)
    identity, dist = bx.query(bits)
    print(f"{img} → {identity} (hamming={dist})")