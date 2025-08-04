# camera_voucher_with_face_box.py

import cv2
import time
import os
import json
import torch
from facenet_pytorch import MTCNN
from voucher_generator import generate_voucher, get_next_voucher_filename

# === Setup paths ===
TEMP_DIR = "./temp_captures"
os.makedirs(TEMP_DIR, exist_ok=True)

# === Load Face Detector (MTCNN) ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(keep_all=False, device=device)

# === Webcam settings ===
cap = cv2.VideoCapture(0)
FRAME_LIMIT = 10
CAPTURE_INTERVAL = 0.5  # seconds between capture attempts

if not cap.isOpened():
    print("❌ Could not access webcam.")
    exit()

print("🎥 Starting webcam feed. Press Q to exit.")

frame_count = 0
saved_count = 0

while frame_count < FRAME_LIMIT:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for MTCNN
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Save only one face frame per interval
        image_path = os.path.join(TEMP_DIR, f"frame_{frame_count}.jpg")
        cv2.imwrite(image_path, frame)

        try:
            voucher = generate_voucher(image_path)
            if voucher:
                filename = get_next_voucher_filename()
                with open(filename, "w") as f:
                    json.dump(voucher, f, indent=2)
                print(f"[✓] Voucher saved: {filename}")
                saved_count += 1
            else:
                print(f"[x] No voucher for frame {frame_count}")
        except Exception as e:
            print(f"[x] Error processing voucher: {e}")

        frame_count += 1
        time.sleep(CAPTURE_INTERVAL)

    else:
        # Optional: display no-face message
        cv2.putText(frame, "No face detected", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # === Show the frame ===
    cv2.imshow("🔴 Live Feed - Press Q to exit", frame)

    # Allow manual exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Done: {saved_count} vouchers saved from {frame_count} frames.")
