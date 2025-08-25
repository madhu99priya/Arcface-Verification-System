import json
import csv
from tqdm import tqdm


INPUT_JSON_FILE = "reduced_dataset2.json"
OUTPUT_CSV_FILE = "review_list2.csv" 
MIN_IMAGES_IN_FOLDER = 3

DISSIMILARITY_THRESHOLD = 0.4 

def hamming_dissimilarity(h1, h2):
    """Calculates the Hamming dissimilarity (ratio of differing bits)."""
    if not h1 or len(h1) != len(h2): return 1.0
    dissimilar_bits = sum(c1 != c2 for c1, c2 in zip(h1, h2))
    return dissimilar_bits / len(h1)


print(f"🔍 Loading dataset from '{INPUT_JSON_FILE}'...")
try:
    with open(INPUT_JSON_FILE, 'r') as f: data = json.load(f)
except FileNotFoundError:
    print(f"❌ Error: The file '{INPUT_JSON_FILE}' was not found.")
    exit()

highly_dissimilar_images = []
print("🕵️  Analyzing folders to find highly dissimilar images...")
for folder_name, images in tqdm(data.items(), desc="Processing Persons", ncols=100):
    if len(images) < MIN_IMAGES_IN_FOLDER: continue

    hashes = [img['hash'] for img in images]
    filenames = [img['filename'] for img in images]

    for i in range(len(hashes)):
        dissimilarities = [hamming_dissimilarity(hashes[i], hashes[j]) for j in range(len(hashes)) if i != j]
        if dissimilarities:
            avg_dissimilarity = sum(dissimilarities) / len(dissimilarities)
        
            if avg_dissimilarity >= DISSIMILARITY_THRESHOLD:
                highly_dissimilar_images.append({
                    "FolderName": folder_name,
                    "FileName": filenames[i],
                    "AvgHammingDissimilarity": avg_dissimilarity
                })

print("📊 Sorting results to show most likely mismatches at the top...")
highly_dissimilar_images.sort(key=lambda x: x['AvgHammingDissimilarity'], reverse=True)

with open(OUTPUT_CSV_FILE, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=["FolderName", "FileName", "AvgHammingDissimilarity"])
    writer.writeheader()
    writer.writerows(highly_dissimilar_images)

print(f"\n✅ Analysis complete! Found {len(highly_dissimilar_images)} highly suspicious images.")
print(f"   Review list has been saved to '{OUTPUT_CSV_FILE}'.")
print("\nNEXT STEP: Run the '4_interactive_review_tool.py' script.")