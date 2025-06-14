import faiss, numpy as np

class BinaryIndex:
    def __init__(self):
        self.index = faiss.IndexBinaryFlat(128)
        self.ids = []

    def add(self, bits: np.ndarray, identity: str):
        self.index.add(np.packbits(bits).reshape(1, -1))
        self.ids.append(identity)

    def query(self, bits: np.ndarray, top_k=1, thresh=15):
        q = np.packbits(bits).reshape(1, -1)
        D, I = self.index.search(q, top_k)
        if D[0][0] <= thresh:
            return self.ids[I[0][0]], D[0][0]
        return None, D[0][0]