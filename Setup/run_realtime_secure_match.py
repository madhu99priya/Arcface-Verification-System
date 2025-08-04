# run_realtime_secure_match.py

import cv2
import os
import time
import json
import torch
from collections import defaultdict
from facenet_pytorch import MTCNN
from voucher_generator import generate_voucher, get_next_voucher_filename
from reconstruct_and_decrypt import reconstruct_and_decrypt
from psi_match_client_server import (
    load_reference_db,
    load_trained_pca,
    load_neuralhash_hyperplanes,
    generate_private_key,
    blind_neuralhash,
    hamming_distance,
    log_event
)
from neural_hash_gen import process_image

# === Configuration ===
TEMP_DIR = "./temp_captures"
VOUCHER_DIR = "./vouchers"
THRESHOLD = 3
CAPTURE_INTERVAL = 0.6
FRAME_LIMIT = 3

# === Setup ===
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(VOUCHER_DIR, exist_ok=True)
cap = cv2.VideoCapture(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

pca = load_trained_pca("./models/pca_512_to_128.pkl")
hyperplanes = load_neuralhash_hyperplanes("./models/neuralhash_128x96_seed1.dat")
private_key = generate_private_key("./models/ecc_private.pem")
reference_db = load_reference_db("./db/reference_blinded_hashes.json")

voucher_groups = defaultdict(list)
frame_count = 0

print("📡 System running. Press Q to stop.")

while frame_count < FRAME_LIMIT:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save and generate voucher
        path = os.path.join(TEMP_DIR, f"frame_{frame_count}.jpg")
        cv2.imwrite(path, frame)

        try:
            voucher = generate_voucher(path)
            if voucher:
                fname = get_next_voucher_filename()
                with open(fname, "w") as f:
                    json.dump(voucher, f, indent=2)
                print(f"[✓] Voucher saved: {fname}")

                bh = voucher["blinded_hash"]
                x = voucher["payload"]["share"]["x"]
                y_hex = voucher["payload"]["share"]["y"]
                y = int(y_hex, 16)
                encrypted = voucher["payload"]["inner_encrypted"]
                header = voucher["header"]

                voucher_groups[bh].append((x, y, encrypted, header))

                # Try reconstruct if we have enough
                if len(voucher_groups[bh]) >= THRESHOLD:
                    reconstruct_and_decrypt(bh, voucher_groups[bh])

                    # === Optional: Run probe match ===
                    result = process_image(path, pca, hyperplanes)
                    probe_bits = result["hash_bits"]
                    probe_hash = result["neural_hash"]
                    blinded = blind_neuralhash(probe_hash, private_key)

                    # Hamming-based lookup
                    for person, entries in reference_db.items():
                        for entry in entries:
                            dist = hamming_distance(probe_bits, entry["hash_bits"])
                            if dist <= 15:
                                log_event("MATCH_FOUND", f"{person} (Hamming={dist})")
                                print(f"✅ MATCH: {person} | Hamming: {dist}")
            else:
                print(f"[x] No voucher generated.")

        except Exception as e:
            print(f"[x] Error processing frame {frame_count}: {e}")

        frame_count += 1
        time.sleep(CAPTURE_INTERVAL)

    else:
        cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("🔴 Secure Match Feed - Press Q to stop", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Capture ended.")
