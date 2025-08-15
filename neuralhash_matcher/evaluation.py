import os
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from neuralhash_utils import load_pca_model, load_hyperplanes, generate_neuralhash
from scipy.spatial.distance import cosine

# ===== Load model =====
pca = load_pca_model('./models/pca_512_to_128.pkl')
hyperplanes = load_hyperplanes('./models/neuralhash_128x96_seed1.dat')

DATASET_DIR = "./data/gallery/val" 

def get_embedding(image_path):
    bits = generate_neuralhash(image_path, pca, hyperplanes)
    return np.array(bits, dtype=np.float32)

# ===== Generate pairs =====
def generate_pairs(dataset_dir):
    persons = os.listdir(dataset_dir)
    pos_pairs, neg_pairs = [], []

    # Positive pairs
    for person in persons:
        imgs = [os.path.join(dataset_dir, person, f) for f in os.listdir(os.path.join(dataset_dir, person))]
        for img1, img2 in itertools.combinations(imgs, 2):
            pos_pairs.append((img1, img2, 1))  # label 1 = genuine

    # Negative pairs
    for p1, p2 in itertools.combinations(persons, 2):
        imgs1 = [os.path.join(dataset_dir, p1, f) for f in os.listdir(os.path.join(dataset_dir, p1))]
        imgs2 = [os.path.join(dataset_dir, p2, f) for f in os.listdir(os.path.join(dataset_dir, p2))]
        for img1 in imgs1:
            for img2 in imgs2:
                neg_pairs.append((img1, img2, 0))  # label 0 = impostor

    return pos_pairs + neg_pairs

pairs = generate_pairs(DATASET_DIR)
print(f"Total pairs: {len(pairs)}")

# ===== Evaluate =====
y_true = []
scores = []

for img1, img2, label in pairs:
    emb1 = get_embedding(img1)
    emb2 = get_embedding(img2)
    sim = 1 - cosine(emb1, emb2)  # cosine similarity
    y_true.append(label)
    scores.append(sim)

y_true = np.array(y_true)
scores = np.array(scores)

# ===== Threshold sweep =====
thresholds = np.linspace(0, 1, 100)
GARs, FPRs = [], []
for th in thresholds:
    y_pred = (scores >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    gar = tp / (tp + fn)
    fpr = fp / (fp + tn)
    GARs.append(gar)
    FPRs.append(fpr)

# ===== ROC Curve =====
fpr_vals, tpr_vals, _ = roc_curve(y_true, scores)
roc_auc = auc(fpr_vals, tpr_vals)

plt.figure()
plt.plot(fpr_vals, tpr_vals, label=f'ROC Curve (AUC={roc_auc:.4f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (GAR)')
plt.title('ROC Curve')
plt.legend()
plt.grid()
plt.show()

# ===== GAR vs FPR plot =====
plt.figure()
plt.plot(FPRs, GARs, marker='o')
plt.xlabel('FPR')
plt.ylabel('GAR')
plt.title('GAR vs FPR')
plt.grid()
plt.show()

# ===== Confusion Matrix at optimal threshold =====
opt_th = thresholds[np.argmax(GARs)]
y_pred_opt = (scores >= opt_th).astype(int)
cm = confusion_matrix(y_true, y_pred_opt)
print(f"Optimal Threshold: {opt_th:.4f}")
print("Confusion Matrix:")
print(cm)
