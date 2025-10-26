import os
import cv2
import json
import numpy as np
import faiss
from detector import SCRFDDetector  # your face detector module
from utils.embedder import Embedder
from enhance_face import enhance_face  # your GFPGAN wrapper

# --- CONFIG ---
STUDENTS_DIR = "face_db/students"
EMB_DIR = "face_db/embeddings"
LOG_FILE = "attendance_logs/attendance.json"
FAISS_INDEX_FILE = os.path.join(EMB_DIR, "faiss_index.bin")
STUDENT_IDS_FILE = os.path.join(EMB_DIR, "student_ids.json")
DEVICE = "cpu"  # or 'cuda'

# --- LOAD FAISS ---
index = faiss.read_index(FAISS_INDEX_FILE)
with open(STUDENT_IDS_FILE, "r") as f:
    student_ids = json.load(f)

# --- INIT MODELS ---
embedder = Embedder(device=DEVICE)
detector = SCRFDDetector()  # must return list of aligned faces per image

# --- LOAD IMAGE ---
import sys
if len(sys.argv) != 2:
    print("Usage: python recognize_multi_face.py <image_path>")
    sys.exit(1)

img_path = sys.argv[1]
img = cv2.imread(img_path)
if img is None:
    raise ValueError(f"Cannot read image {img_path}")

# --- DETECT FACES ---
faces = detector.get_aligned_faces(img)
attendance = []

for face in faces:
    # --- ENHANCE IF LOW-QUALITY ---
    enhanced_face = enhance_face(face)
    
    # --- GET EMBEDDING ---
    emb = embedder.get_embedding(enhanced_face).reshape(1, -1)
    
    # --- SEARCH FAISS ---
    D, I = index.search(emb, k=1)
    student_id = student_ids[I[0][0]]
    score = D[0][0]
    
    attendance.append({"student_id": student_id, "score": float(score)})

# --- SAVE ATTENDANCE ---
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, "r") as f:
        logs = json.load(f)
else:
    logs = []

logs.append({
    "image": img_path,
    "attendance": attendance
})

with open(LOG_FILE, "w") as f:
    json.dump(logs, f, indent=2)

print(f"Attendance recorded for {len(attendance)} students in {img_path}")
