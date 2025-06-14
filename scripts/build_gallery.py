import os
from utils.align import align_face
from utils.embed import get_embedding
from utils.hash import compute_neural_hash
from utils.index import BinaryIndex
import faiss

bx = BinaryIndex()
for person in os.listdir('data/gallery'):
    folder = os.path.join('data/gallery', person)
    for img in os.listdir(folder):
        aligned = align_face(os.path.join(folder, img))
        if aligned is None: continue
        emb = get_embedding(aligned)
        bits = compute_neural_hash(emb)
        bx.add(bits, person)

# Save index & ids for later (optional)
faiss.write_index(bx.index, 'models/gallery.index')
with open('models/gallery.ids', 'w') as f:
    f.write("\n".join(bx.ids))