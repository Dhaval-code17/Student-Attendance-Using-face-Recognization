import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from gfpgan import GFPGANer
from insightface.app import FaceAnalysis
import faiss

# ---- PATH CONFIG ----
BASE_DIR = r"D:\SMART ATTENDANCE"
STUDENTS_DIR = os.path.join(BASE_DIR, "face_db", "students")
EMB_DIR = os.path.join(BASE_DIR, "face_db", "embeddings")
FAISS_INDEX_PATH = os.path.join(EMB_DIR, "faiss_index.bin")
STUDENT_IDS_PATH = os.path.join(EMB_DIR, "student_ids.json")

# ---- LOAD MODELS ----
print("[INFO] Initializing GFPGAN and ArcFace ...")
gfpganer = GFPGANer(
    model_path="GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth",
    upscale=1,
    arch="clean",
    channel_multiplier=2,
    bg_upsampler=None
)

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

# ---- LOAD STUDENT REGISTRY ----
if os.path.exists(STUDENT_IDS_PATH):
    with open(STUDENT_IDS_PATH, "r") as f:
        student_ids = json.load(f)
else:
    student_ids = []

index = faiss.read_index(FAISS_INDEX_PATH) if os.path.exists(FAISS_INDEX_PATH) else None

# ---- PARAMETERS ----
QUALITY_THRESHOLD = 0.6  # faces below this score will be enhanced

# ---- PROCESS ALL STUDENTS ----
for sid in student_ids:
    print(f"\n[STUDENT] {sid}")
    student_dir = os.path.join(STUDENTS_DIR, sid)
    aligned_dir = os.path.join(student_dir, "aligned")
    enhanced_dir = os.path.join(student_dir, "enhanced")
    os.makedirs(enhanced_dir, exist_ok=True)

    qfile = os.path.join(student_dir, "face_quality_scores.json")
    if not os.path.exists(qfile):
        print(f"  ⚠ No quality file for {sid}")
        continue

    with open(qfile, "r") as f:
        quality_data = json.load(f)

    # ---- LOOP THROUGH FACES ----
    new_embeddings = []

    # Updated: handle JSON as dictionary {filename: score}
    for img_name, qdata in tqdm(quality_data.items(), desc=f"Enhancing {sid}"):
        src_path = os.path.join(aligned_dir, img_name)
        dst_path = os.path.join(enhanced_dir, img_name)

    # Extract actual quality score
        qscore = qdata.get("score", 1.0)  # default 1.0 if missing

    # Skip high-quality faces
        if qscore >= QUALITY_THRESHOLD:
           continue


        try:
            face = cv2.imread(src_path)
            if face is None:
                print(f"  ⚠ Skipping unreadable {img_name}")
                continue

            # Enhance
            _, restored_faces, _ = gfpganer.enhance(
                face, has_aligned=True, only_center_face=True, paste_back=False
            )
            enhanced = restored_faces[0]
            cv2.imwrite(dst_path, enhanced)

            # Recompute embedding
            faces = app.get(enhanced)
            if faces:
                emb = faces[0].embedding.astype(np.float32)
                new_embeddings.append(emb)

        except Exception as e:
            print(f"  ❌ Error on {img_name}: {e}")

    # ---- UPDATE FAISS INDEX ----
    if new_embeddings:
        print(f"  [+] {len(new_embeddings)} enhanced embeddings added for {sid}")
        if index is None:
            index = faiss.IndexFlatIP(new_embeddings[0].shape[0])
        faiss.normalize_L2(new_embeddings)
        index.add(np.array(new_embeddings, dtype=np.float32))

# ---- SAVE UPDATED INDEX ----
if index is not None:
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"\n[INFO] Updated FAISS index saved → {FAISS_INDEX_PATH}")
else:
    print("\n⚠ No new embeddings to update.")
