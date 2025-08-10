import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.special import ndtri
from tqdm import tqdm

from neural_hash import generate_neuralhash, load_pca_model, load_hyperplanes

# Config
DATASET_DIR = r"F:\\Dataset_VGGFace2\\train"
DAT_FILE = r"./models/neuralhash_128x96_seed1.dat"
PCA_MODEL_FILE = r"./models/pca_512_to_128.pkl"
MAX_IMAGES_PER_PERSON = 20

# Load models
pca = load_pca_model(PCA_MODEL_FILE)
hyperplanes = load_hyperplanes(DAT_FILE)

# Hamming distance function
def hamming_distance(bits1, bits2):
    return np.count_nonzero(bits1 != bits2)

# Prepare scores
genuine_scores = []
impostor_scores = []

persons = sorted([p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p))])
total = len(persons)

for idx, person in enumerate(tqdm(persons, desc="Processing persons")):
    person_path = os.path.join(DATASET_DIR, person)
    images = sorted(os.listdir(person_path))[:MAX_IMAGES_PER_PERSON]

    if len(images) < 2:
        continue

    try:
        probe_hash = generate_neuralhash(os.path.join(person_path, images[0]), pca, hyperplanes)
    except:
        continue

    # Genuine comparisons (same person)
    for img in images[1:]:
        try:
            compare_hash = generate_neuralhash(os.path.join(person_path, img), pca, hyperplanes)
            dist = hamming_distance(probe_hash, compare_hash) / 128.0
            genuine_scores.append(dist)
        except:
            continue

    # Impostor comparisons
    for jdx, other_person in enumerate(persons):
        if other_person == person:
            continue
        other_path = os.path.join(DATASET_DIR, other_person)
        other_images = sorted(os.listdir(other_path))[:MAX_IMAGES_PER_PERSON]

        for other_img in other_images:
            try:
                other_hash = generate_neuralhash(os.path.join(other_path, other_img), pca, hyperplanes)
                dist = hamming_distance(probe_hash, other_hash) / 128.0
                impostor_scores.append(dist)
            except:
                continue

# Evaluation
y_true = [0] * len(genuine_scores) + [1] * len(impostor_scores)
y_scores = genuine_scores + impostor_scores

fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
fnr = 1 - tpr
roc_auc = auc(fpr, tpr)

eer_index = np.nanargmin(np.abs(fpr - fnr))
eer = fpr[eer_index]
eer_threshold = thresholds[eer_index]

precision, recall, _ = precision_recall_curve(y_true, [-s for s in y_scores])
ap_score = average_precision_score(y_true, [-s for s in y_scores])

acc = []
for thresh in thresholds:
    preds = [1 if s >= thresh else 0 for s in y_scores]
    correct = np.sum(np.array(preds) == np.array(y_true))
    acc.append(correct / len(y_true))

# Create output directory
os.makedirs("evaluation_plots", exist_ok=True)

# ROC Curve
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate (FAR)")
plt.ylabel("True Positive Rate (1 - FRR)")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/roc_curve.png")
plt.close()

# DET Curve
plt.figure(figsize=(6, 5))
plt.plot(ndtri(fpr), ndtri(fnr), label='DET Curve')
plt.xlabel("False Positive Rate (FAR)")
plt.ylabel("False Negative Rate (FRR)")
plt.title("DET Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/det_curve.png")
plt.close()

# Precision-Recall Curve
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f'AP = {ap_score:.4f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/precision_recall_curve.png")
plt.close()

# Accuracy vs Threshold
plt.figure(figsize=(6, 5))
plt.plot(thresholds, acc)
plt.axvline(eer_threshold, color='red', linestyle='--', label=f'EER Threshold = {eer_threshold:.4f}')
plt.title("Accuracy vs Threshold")
plt.xlabel("Threshold (Hamming Distance)")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/accuracy_vs_threshold.png")
plt.close()

# FAR & FRR vs Threshold
plt.figure(figsize=(6, 5))
plt.plot(thresholds, fpr, label='FAR (FPR)')
plt.plot(thresholds, fnr, label='FRR (FNR)')
plt.axvline(eer_threshold, color='red', linestyle='--', label='EER Threshold')
plt.title("FAR and FRR vs Threshold")
plt.xlabel("Threshold (Hamming Distance)")
plt.ylabel("Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/far_frr_vs_threshold.png")
plt.close()

# Combined Metrics Plot
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

axs[0, 0].plot(fpr, tpr)
axs[0, 0].plot([0, 1], [0, 1], 'k--')
axs[0, 0].set_title("ROC Curve")
axs[0, 0].set_xlabel("FAR")
axs[0, 0].set_ylabel("1 - FRR")
axs[0, 0].grid(True)

axs[0, 1].plot(ndtri(fpr), ndtri(fnr))
axs[0, 1].set_title("DET Curve")
axs[0, 1].set_xlabel("FAR")
axs[0, 1].set_ylabel("FRR")
axs[0, 1].grid(True)

axs[1, 0].plot(recall, precision)
axs[1, 0].set_title("Precision-Recall")
axs[1, 0].set_xlabel("Recall")
axs[1, 0].set_ylabel("Precision")
axs[1, 0].grid(True)

axs[1, 1].plot(thresholds, acc)
axs[1, 1].axvline(eer_threshold, color='red', linestyle='--')
axs[1, 1].set_title("Accuracy vs Threshold")
axs[1, 1].set_xlabel("Threshold")
axs[1, 1].set_ylabel("Accuracy")
axs[1, 1].grid(True)

fig.suptitle("Face Recognition System Evaluation Metrics", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("evaluation_plots/combined_metrics.png")
plt.close()

print("✅ All plots saved in: evaluation_plots/")
print(f"Total Genuine Comparisons: {len(genuine_scores)}")
print(f"Total Impostor Comparisons: {len(impostor_scores)}")
print(f"Equal Error Rate (EER): {eer:.4f}")
print(f"EER Threshold: {eer_threshold:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Average Precision Score (PR Curve): {ap_score:.4f}")
