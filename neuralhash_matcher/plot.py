# File: 2_plot_results_final.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import auc
import numpy as np
import os

output_dir = "..\\evaluation_plots"
os.makedirs(output_dir, exist_ok=True) 

# --- 1. Load the Metrics Data ---
csv_file = "../evaluation_results/evaluation_metrics_2_detailed_hamming_only.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print(f"Error: '{csv_file}' not found.")
    print("Please run the '1_generate_and_analyze_metrics_2_hamming_only.py' script first.")
    exit()

# --- 2. Find the Equal Error Rate (EER) ---
# EER is the point where FAR is approximately equal to FRR.
# We find the threshold where the absolute difference between FAR and FRR is smallest.
eer_index = abs(df['FAR_FPR'] - df['FRR']).idxmin()
eer_threshold = df.loc[eer_index, 'Threshold']
# The EER value is the average of FAR and FRR at this point
eer_value = (df.loc[eer_index, 'FAR_FPR'] + df.loc[eer_index, 'FRR']) / 2

print("\n--- System Performance Summary ---")
print(f"Equal Error Rate (EER) is {eer_value:.4f} (or {eer_value:.2%})")
print(f"This occurs at a threshold of ~{eer_threshold:.2f}")
print("------------------------------------")

#What is EER?
#EER stands for Equal Error Rate.
#In simple terms, it is the single point on the graph where the system's
# two main types of errors are equal. It represents the "perfectly balanced" trade-off between security and convenience.

# --- 3. Generate and Save Plot 1: ROC Curve ---
# AUC (Area Under Curve) is a great single-metric summary of the ROC curve's performance.
# Note: The AUC requires the FPR values to be sorted.
df_sorted = df.sort_values(by='FAR_FPR')
roc_auc = auc(df_sorted['FAR_FPR'], df_sorted['GAR_TPR'])

plt.figure(figsize=(8, 6))
plt.plot(df['FAR_FPR'], df['GAR_TPR'], color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FAR / FPR)')
plt.ylabel('True Positive Rate (GAR / TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
roc_filename = os.path.join(output_dir, "1_roc_curve3.png")
plt.savefig(roc_filename)
print(f"✅ ROC Curve saved to {roc_filename}")


# --- 4. Generate and Save Plot 2: DET Curve ---
plt.figure(figsize=(8, 8))
plt.plot(df['FAR_FPR'], df['FRR'], color='darkred', lw=2, label='DET Curve')
plt.plot(eer_value, eer_value, 'o', color='blue', markersize=8, label=f'EER = {eer_value:.4f}')
plt.xscale('log')
plt.yscale('log')
# Set axis limits to be slightly outside the data range for better visualization
plt.xlim([max(1e-5, df['FAR_FPR'][df['FAR_FPR']>0].min() * 0.5), 1.0])
plt.ylim([max(1e-5, df['FRR'][df['FRR']>0].min() * 0.5), 1.0])
plt.xlabel('False Acceptance Rate (FAR)')
plt.ylabel('False Rejection Rate (FRR)')
plt.title('Detection Error Trade-off (DET) Curve')
plt.legend(loc="upper right")
plt.grid(True, which="both", ls="--")
det_filename = os.path.join(output_dir, "2_det_curve3.png")
plt.savefig(det_filename)
print(f"✅ DET Curve saved to {det_filename}")


# --- 5. Generate and Save Plot 3: Error Rate vs. Threshold Curve ---
plt.figure(figsize=(10, 7))
plt.plot(df['Threshold'], df['FAR_FPR'], lw=2, label='False Acceptance Rate (FAR)', color='red')
plt.plot(df['Threshold'], df['FRR'], lw=2, label='False Rejection Rate (FRR)', color='blue')
# Add a point and annotation for the EER
plt.plot(eer_threshold, eer_value, 'o', color='black', markersize=10, label=f'EER Point (~{eer_value:.4f})')
plt.annotate(f'Balanced Threshold\n(EER ≈ {eer_value:.2%})',
             xy=(eer_threshold, eer_value),
             xytext=(eer_threshold, eer_value * 5), # Adjust text position
             arrowprops=dict(facecolor='black', shrink=0.05),
             ha='center')
plt.yscale('log') # Log scale is best for viewing error rates
plt.xlabel("Similarity Threshold")
plt.ylabel("Error Rate (Log Scale)")
plt.title("Error Rate vs. Similarity Threshold")
plt.legend()
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
threshold_filename = os.path.join(output_dir, "3_error_vs_threshold3.png")
plt.savefig(threshold_filename)
print(f"✅ Error Rate vs. Threshold plot saved to {threshold_filename}")

plt.show() # Display all generated plots