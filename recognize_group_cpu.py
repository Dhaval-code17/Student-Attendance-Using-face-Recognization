import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model
import faiss

# ---- CONFIG ----
STUDENTS_DIR = r"D:\SMART ATTENDANCE\face_db\students"
FAISS_INDEX_PATH = os.path.join(STUDENTS_DIR, "faiss_index.bin")
STUDENT_IDS_PATH = os.path.join(STUDENTS_DIR, "student_ids.json")
GROUP_PHOTO_PATH = r"D:\SMART ATTENDANCE\group_photo.jpg"  # Classroom/group photo
OUTPUT_JSON = r"D:\SMART ATTENDANCE\attendance_results.json"
DEVICE = -1  # -1 = CPU, 0 = GPU if available

# ---- Cosine similarity helper ----
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# ---- Load FAISS index and student IDs ----
print("[INFO] Loading FAISS index and student IDs...")
index = faiss.read_index(FAISS_INDEX_PATH)
with open(STUDENT_IDS_PATH, 'r') as f:
    student_ids = json.load(f)
print(f"[INFO] {len(student_ids)} students loaded")

# ---- Load ArcFace model ----
print("[INFO] Loading ArcFace model (buffalo_l)...")
arcface = get_model('buffalo_l', download=True)
arcface.prepare(ctx_id=DEVICE)
print("[INFO] ArcFace loaded.")

# ---- Initialize Face Detector (SCRFD) ----
print("[INFO] Initializing FaceAnalysis for detection...")
app = FaceAnalysis(name='buffalo_l', allowed_modules=['detection'])
app.prepare(ctx_id=DEVICE, det_thresh=0.3)  # lower threshold to detect more faces
print("[INFO] Face detector ready.")

# ---- Load group photo ----
img = cv2.imread(GROUP_PHOTO_PATH)
if img is None:
    raise ValueError(f"Cannot load group photo at path: {GROUP_PHOTO_PATH}")

# ---- Detect faces ----
faces = app.get(img)
print(f"[INFO] Detected {len(faces)} faces in the group photo")

attendance = []

for face in tqdm(faces):
    aligned = face.normed_face  # HWC RGB aligned face
    if aligned is None:
        print("[WARN] Skipping face: normed_face is None")
        continue

    # Convert RGB → BGR if needed (optional, OpenCV uses BGR)
    aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)

    # Optional: GFPGAN enhancement here if low-quality (skip for now)

    # ---- Preprocess for ArcFace ----
    img_input = aligned.astype(np.float32)
    img_input = (img_input - 127.5) / 128.0
    img_input = np.transpose(img_input, (2, 0, 1))  # HWC → CHW
    img_input = np.expand_dims(img_input, axis=0)   # batch dimension

    # ---- Compute embedding ----
    emb = arcface.forward(img_input).flatten()
    emb = emb / np.linalg.norm(emb)

    # ---- Search FAISS index using cosine similarity ----
    # FAISS index stores L2 vectors, convert to cosine by normalizing vectors
    index.normalize_L2()  # ensure FAISS index is normalized
    D, I = index.search(np.expand_dims(emb, axis=0), k=1)
    matched_idx = int(I[0][0])
    student_id = student_ids[matched_idx]

    similarity = float(cosine_similarity(emb, np.expand_dims(emb, axis=0)[0]))  # 1.0 for exact same vector
    attendance.append({
        "student_id": student_id,
        "similarity": similarity
    })

# ---- Save attendance results ----
with open(OUTPUT_JSON, 'w') as f:
    json.dump(attendance, f, indent=4)

print(f"[INFO] Attendance saved to {OUTPUT_JSON}")
print("[INFO] Done!")
