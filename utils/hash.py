import onnxruntime as ort
import numpy as np
from PIL import Image
import os

class NeuralHasher:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.session = ort.InferenceSession(model_path)

    def preprocess(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert('RGB').resize((360, 360))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0  # Normalize to [-1, 1]
        arr = arr.transpose(2, 0, 1).reshape(1, 3, 360, 360)
        return arr

    def get_hash(self, image_path: str) -> np.ndarray:
        input_tensor = self.preprocess(image_path)
        input_name = self.session.get_inputs()[0].name
        output = self.session.run(None, {input_name: input_tensor})[0].flatten()
        bits = np.array([1 if v >= 0 else 0 for v in output], dtype=np.uint8)
        return bits  # shape: (128,)

    def hamming_distance(self, h1: np.ndarray, h2: np.ndarray) -> int:
        return np.count_nonzero(h1 != h2)


# === Test Your Images ===

if __name__ == "__main__":
    model_path = "models/model.onnx"
    image_1 = "data/probe/image4.jpg"
    image_2 = "data/probe/image5.jpg"

    hasher = NeuralHasher(model_path)
    hash1 = hasher.get_hash(image_1)
    hash2 = hasher.get_hash(image_2)

    hamming = hasher.hamming_distance(hash1, hash2)
    similarity = 100 - (hamming / 128 * 100)

    print(f"\n🔍 Comparing: {os.path.basename(image_1)} vs {os.path.basename(image_2)}")
    print(f"→ Hamming Distance: {hamming}")
    print(f"→ Similarity Score: {similarity:.2f}%")

    threshold = 15  # or experiment with values
    if hamming <= threshold:
        print("✅ Likely same person (based on NeuralHash similarity).")
    else:
        print("❌ Likely different (NeuralHash not close enough).")
