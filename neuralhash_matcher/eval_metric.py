# File: 1_generate_and_analyze_metrics_hamming_only.py

import json
import numpy as np
from itertools import combinations
from tqdm import tqdm
import warnings
import csv
import random
import os

warnings.simplefilter(action='ignore', category=FutureWarning)
random.seed(42)  # for reproducibility

output_dir = "..\\evaluation_results"
os.makedirs(output_dir, exist_ok=True)

def hamming_similarity(h1, h2):
    """Calculates the Hamming similarity between two binary hash strings."""
    dissimilar_bits = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    return 1 - (dissimilar_bits / 96)  # assuming hash length = 96 bits

# ===== 1. Load and Pre-process Data =====
print("📥 Loading pre-computed embeddings from reduced_dataset.json...")
with open("reduced_dataset2.json", "r") as f:
    data = json.load(f)

# ===== 2. Collect Genuine and Impostor Pairs =====
genuine_scores = []
impostor_scores = []

# Collect scores for same-person pairs (genuine)
print("🔍 Calculating scores for genuine (same-person) pairs...")
for person, imgs in tqdm(data.items(), ncols=100, unit="person"):
    if len(imgs) < 2:
        continue
    for img1_data, img2_data in combinations(imgs, 2):
        h1 = img1_data["hash"]
        h2 = img2_data["hash"]
        sim = hamming_similarity(h1, h2)
        genuine_scores.append(sim)

# Collect scores for different-person pairs (impostor)
print("🚫 Calculating scores for impostor (different-person) pairs...")
persons = list(data.keys())
for p1, p2 in tqdm(combinations(persons, 2), ncols=100, unit="pair", total=len(persons)*(len(persons)-1)//2):
    if data[p1] and data[p2]:
        h1 = data[p1][0]["hash"]
        h2 = data[p2][0]["hash"]
        sim = hamming_similarity(h1, h2)
        impostor_scores.append(sim)

# ===== 3. Balance Genuine and Impostor Pairs =====
num_pairs = min(len(genuine_scores), len(impostor_scores))
print(f"\nBalancing pairs: Using {num_pairs} genuine and {num_pairs} impostor pairs for evaluation.")

genuine_scores_sampled = random.sample(genuine_scores, num_pairs)
impostor_scores_sampled = random.sample(impostor_scores, num_pairs)

# ===== 4. Sweep Thresholds and Calculate Metrics =====
thresholds = np.arange(0, 1.00, 0.01)
results = []

num_genuine = len(genuine_scores_sampled)
num_impostor = len(impostor_scores_sampled)

print(f"\nTotal Genuine Pairs Used: {num_genuine}")
print(f"Total Impostor Pairs Used: {num_impostor}\n")

print("📈 Evaluating thresholds to calculate metrics...")
for t in tqdm(thresholds, ncols=100):
    tp = sum(1 for score in genuine_scores_sampled if score >= t)
    fp = sum(1 for score in impostor_scores_sampled if score >= t)
    fn = num_genuine - tp
    tn = num_impostor - fp
    
    gar = tp / num_genuine if num_genuine else 0
    far = fp / num_impostor if num_impostor else 0
    frr = fn / num_genuine if num_genuine else 0
    
    results.append({
        "Threshold": round(t, 2),
        "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "GAR_TPR": gar, "FAR_FPR": far, "FRR": frr
    })

# ===== 5. Save Results to CSV =====
csv_file = os.path.join(output_dir, "evaluation_metrics_2_detailed_hamming_only.csv")
fieldnames = ["Threshold", "TP", "FN", "FP", "TN", "GAR_TPR", "FAR_FPR", "FRR"]

with open(csv_file, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Evaluation complete! Detailed results saved to {csv_file}")
