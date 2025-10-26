import cv2
import faiss
import json
import numpy as np
import os
from datetime import datetime
from insightface.app import FaceAnalysis

# --- Path setup ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
EMB_DIR = os.path.join(BASE_DIR, "face_db", "embeddings")
ATTEND_DIR = os.path.join(BASE_DIR, "attendance_logs")

os.makedirs(ATTEND_DIR, exist_ok=True)

# --- Load FAISS index and student list ---
index_path = os.path.join(EMB_DIR, "faiss_index.bin")
id_json_path = os.path.join(EMB_DIR, "student_ids.json")

if not os.path.exists(index_path) or not os.path.exists(id_json_path):
    raise FileNotFoundError("❌ Missing FAISS index or student list. Please enroll first.")

index = faiss.read_index(index_path)
with open(id_json_path, "r") as f:
    student_ids = json.load(f)

print("[INFO] Loaded FAISS index and student list successfully.")

# --- Initialize model ---
model = FaceAnalysis(name="buffalo_l")
model.prepare(ctx_id=0)
print("[INFO] ArcFace model ready.")

# --- Open webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Unable to access webcam.")

print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[WARN] Frame not captured.")
        continue

    faces = model.get(frame)
    for face in faces:
        emb = face.normed_embedding.astype("float32").reshape(1, -1)

        # --- Find nearest student ---
        D, I = index.search(emb, 1)
        distance = float(D[0][0])
        idx = int(I[0][0])

        if idx < len(student_ids):
            student_id = student_ids[idx]
            name_text = f"{student_id} ({distance:.3f})"

            # Mark attendance if below threshold
            if distance < 1.2:
                log_path = os.path.join(ATTEND_DIR, "attendance.json")

                if os.path.exists(log_path):
                    with open(log_path, "r") as f:
                        logs = json.load(f)
                else:
                    logs = {}

                today = datetime.now().strftime("%Y-%m-%d")
                logs.setdefault(today, {})

                if student_id not in logs[today]:
                    logs[today][student_id] = datetime.now().strftime("%H:%M:%S")
                    with open(log_path, "w") as f:
                        json.dump(logs, f, indent=4)
                    print(f"[ATTENDANCE] {student_id} marked at {logs[today][student_id]}")

            else:
                name_text = "Unknown"

            # --- Draw box + label ---
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, name_text, (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("SMART ATTENDANCE - Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Recognition stopped.")
