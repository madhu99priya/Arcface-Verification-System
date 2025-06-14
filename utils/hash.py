import onnxruntime as ort
import numpy as np

# Load ONNX model
sess = ort.InferenceSession('models/model.onnx')

def compute_neural_hash(embedding: np.ndarray) -> np.ndarray:
    # embedding: (512,) float32
    inp = embedding.astype(np.float32).reshape(1, -1)
    out = sess.run(None, {'input': inp})[0]  # e.g. shape (1,16) bytes
    # Convert to bit array
    bits = np.unpackbits(out.astype(np.uint8), axis=1)
    return bits.squeeze()  # (128,) binary array

# Optional: pack to bytes for Faiss
pack = lambda bits: np.packbits(bits).reshape(1, -1)