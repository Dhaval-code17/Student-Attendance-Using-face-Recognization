import os
import json
import numpy as np
import argparse
import faiss
from insightface.app import FaceAnalysis
import cv2

# ✅ Always use absolute paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "face_db", "students")
EMB_DIR = os.path.join(BASE_DIR, "face_db", "embeddings")

# ✅ Create folders if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EMB_DIR, exist_ok=True)

print("[DEBUG] BASE_DIR =", BASE_DIR)
print("[DEBUG] DATA_DIR =", DATA_DIR)
print("[DEBUG] EMB_DIR  =", EMB_DIR)


def compute_embeddings(student_id):
    print(f"\n[INFO] Starting enrollment for student: {student_id}")

    # Initialize ArcFace model
    model = FaceAnalysis(name='buffalo_l')
    model.prepare(ctx_id=0)

    student_path = os.path.join(DATA_DIR, student_id)
    if not os.path.exists(student_path):
        raise FileNotFoundError(f"[ERROR] Student folder not found: {student_path}")

    embeddings = []

    for img in os.listdir(student_path):
        if img.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(student_path, img)
            frame = cv2.imread(img_path)

            if frame is None:
                print(f"[WARN] Could not read image: {img_path}")
                continue

            faces = model.get(frame)
            if len(faces) == 0:
                print(f"[WARN] No face found in {img}")
                continue

            embeddings.append(faces[0].normed_embedding)

    if len(embeddings) == 0:
        raise ValueError("[ERROR] No valid embeddings found.")

    emb_array = np.vstack(embeddings).astype("float32")

    # ✅ Save embeddings file
    npy_path = os.path.join(EMB_DIR, f"{student_id}_embeddings.npy")
    np.save(npy_path, emb_array)
    print(f"[INFO] Saved embeddings file: {npy_path}")

    # ✅ Create or update FAISS index
    index_path = os.path.join(EMB_DIR, "faiss_index.bin")
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        index = faiss.IndexFlatL2(512)

    index.add(emb_array)
    faiss.write_index(index, index_path)
    print(f"[INFO] Updated FAISS index: {index_path}")

    # ✅ Update student ID list
    id_json = os.path.join(EMB_DIR, "student_ids.json")
    if os.path.exists(id_json):
        with open(id_json, "r") as f:
            student_ids = json.load(f)
    else:
        student_ids = []

    if student_id not in student_ids:
        student_ids.append(student_id)
    with open(id_json, "w") as f:
        json.dump(student_ids, f, indent=4)
    print(f"[INFO] Updated student list: {id_json}")

    # ✅ Verify everything exists
    print("\n[VERIFY]")
    print("[✓]", "Embeddings exist:", os.path.exists(npy_path))
    print("[✓]", "FAISS index exists:", os.path.exists(index_path))
    print("[✓]", "Student list exists:", os.path.exists(id_json))

    print("\n[✅] Enrollment completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    compute_embeddings(args.id)
