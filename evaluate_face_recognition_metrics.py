import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.special import ndtri
from tqdm import tqdm
from PCA_neural_hash import process_image, load_trained_pca, load_hyperplanes, hamming_distance

# Config
DATASET_DIR = r"E:\Accedemic\FYP\new_dataset"  # <-- your dataset folder
DAT_FILE = r"./models/neuralhash_128x96_seed1.dat"
MAX_IMAGES_PER_PERSON = 20

# Load models
pca = load_trained_pca()
hyperplanes = load_hyperplanes(DAT_FILE)

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

    probe_path = os.path.join(person_path, images[0])
    probe_result = process_image(probe_path, pca, hyperplanes)
    if not probe_result['success']:
        continue

    probe_hash = probe_result['hash_bits']

    # Genuine comparisons (same person)
    for img in images[1:]:
        compare_path = os.path.join(person_path, img)
        result = process_image(compare_path, pca, hyperplanes)
        if result['success']:
            dist = hamming_distance(probe_hash, result['hash_bits']) / 128.0
            genuine_scores.append(dist)

    # Impostor comparisons (different persons)
    for jdx, other_person in enumerate(persons):
        if other_person == person:
            continue

        other_path = os.path.join(DATASET_DIR, other_person)
        other_images = sorted(os.listdir(other_path))[:MAX_IMAGES_PER_PERSON]

        for other_img in other_images:
            compare_path = os.path.join(other_path, other_img)
            result = process_image(compare_path, pca, hyperplanes)
            if result['success']:
                dist = hamming_distance(probe_hash, result['hash_bits']) / 128.0
                impostor_scores.append(dist)

# Convert to labels
y_true = [0] * len(genuine_scores) + [1] * len(impostor_scores)
y_scores = genuine_scores + impostor_scores

# Compute ROC and AUC
fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
fnr = 1 - tpr
roc_auc = auc(fpr, tpr)

# Compute EER
eer_index = np.nanargmin(np.abs(fpr - fnr))
eer = fpr[eer_index]
eer_threshold = thresholds[eer_index]

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, [-s for s in y_scores])  # invert scores
ap_score = average_precision_score(y_true, [-s for s in y_scores])

# Accuracy vs Threshold
acc = []
for thresh in thresholds:
    preds = [1 if s >= thresh else 0 for s in y_scores]
    correct = np.sum(np.array(preds) == np.array(y_true))
    acc.append(correct / len(y_true))

# # ===== Plotting =====
# plt.figure(figsize=(6,5))
# plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
# plt.plot([0,1], [0,1], 'k--')
# plt.xlabel("False Positive Rate (FAR)")
# plt.ylabel("True Positive Rate (1 - FRR)")
# plt.title("ROC Curve")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6,5))
# plt.plot(ndtri(fpr), ndtri(fnr), label='DET Curve')
# plt.xlabel("False Positive Rate (FAR)")
# plt.ylabel("False Negative Rate (FRR)")
# plt.title("DET Curve")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6,5))
# plt.plot(recall, precision, label=f'PR Curve (AP = {ap_score:.4f})')
# plt.xlabel("Recall")
# plt.ylabel("Precision")
# plt.title("Precision-Recall Curve")
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(6,5))
# plt.plot(thresholds, acc)
# plt.axvline(eer_threshold, color='red', linestyle='--', label=f'EER Threshold = {eer_threshold:.4f}')
# plt.title("Accuracy vs Threshold")
# plt.xlabel("Threshold (Hamming Distance)")
# plt.ylabel("Accuracy")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# # ===== Summary =====
# print("\n======== Performance Metrics ========")
# print(f"Total Genuine Comparisons: {len(genuine_scores)}")
# print(f"Total Impostor Comparisons: {len(impostor_scores)}")
# print(f"Equal Error Rate (EER): {eer:.4f}")
# print(f"EER Threshold: {eer_threshold:.4f}")
# print(f"ROC AUC: {roc_auc:.4f}")
# print(f"Average Precision Score (PR Curve): {ap_score:.4f}")


# Create output directory
os.makedirs("evaluation_plots", exist_ok=True)

# ----- ROC Curve -----
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

# ----- DET Curve -----
plt.figure(figsize=(6, 5))
plt.plot(ndtri(fpr), ndtri(fnr), label='DET Curve')
plt.xlabel("False Positive Rate (FAR)")
plt.ylabel("False Negative Rate (FRR)")
plt.title("DET Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/det_curve.png")
plt.close()

# ----- Precision-Recall Curve -----
plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f'AP = {ap_score:.4f}')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.tight_layout()
plt.savefig("evaluation_plots/precision_recall_curve.png")
plt.close()

# ----- Accuracy vs Threshold -----
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

# ----- FPR, FNR vs Threshold -----
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

# ----- Combined Plot -----
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# ROC
axs[0, 0].plot(fpr, tpr)
axs[0, 0].plot([0, 1], [0, 1], 'k--')
axs[0, 0].set_title("ROC Curve")
axs[0, 0].set_xlabel("FAR")
axs[0, 0].set_ylabel("1 - FRR")
axs[0, 0].grid(True)

# DET
axs[0, 1].plot(ndtri(fpr), ndtri(fnr))
axs[0, 1].set_title("DET Curve")
axs[0, 1].set_xlabel("FAR")
axs[0, 1].set_ylabel("FRR")
axs[0, 1].grid(True)

# PR Curve
axs[1, 0].plot(recall, precision)
axs[1, 0].set_title("Precision-Recall")
axs[1, 0].set_xlabel("Recall")
axs[1, 0].set_ylabel("Precision")
axs[1, 0].grid(True)

# Accuracy vs Threshold
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
