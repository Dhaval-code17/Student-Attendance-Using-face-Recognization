import os
import numpy as np
import cv2
from insightface.model_zoo import get_model

STUDENTS_DIR = r"D:\SMART ATTENDANCE\face_db\students"

# Load ArcFace
model = get_model('buffalo_l', download=True)
model.prepare(ctx_id=-1)  # CPU

for student in os.listdir(STUDENTS_DIR):
    student_path = os.path.join(STUDENTS_DIR, student)
    aligned_dir = os.path.join(student_path, "aligned")
    if not os.path.exists(aligned_dir):
        continue

    for img_file in os.listdir(aligned_dir):
        if img_file.lower().endswith(".jpg") or img_file.lower().endswith(".png"):
            img_path = os.path.join(aligned_dir, img_file)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # preprocess
            img_rgb = img_rgb.astype(np.float32)
            img_rgb = (img_rgb - 127.5) / 128.0
            img_rgb = np.transpose(img_rgb, (2,0,1))
            img_rgb = np.expand_dims(img_rgb, 0)
            
            # embedding
            emb = model.forward(img_rgb).flatten()
            emb = emb / np.linalg.norm(emb)

            # save embedding
            emb_path = os.path.join(student_path, "arcface_emb.npy")
            np.save(emb_path, emb)
            print(f"[INFO] Saved embedding for {student}: {img_file}")
