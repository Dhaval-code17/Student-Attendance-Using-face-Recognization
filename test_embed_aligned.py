import cv2
import numpy as np
from insightface.model_zoo import get_model

print("[INFO] Loading ArcFace model (buffalo_l) ...")
model = get_model('buffalo_l', download=True)
model.prepare(ctx_id=-1)  # -1 = CPU mode
print("[INFO] Model loaded successfully.")

# ---- Load aligned face ----
img_path = r"D:\SMART ATTENDANCE\face_db\students\23CS002\aligned\aligned_img_01.jpg"
print(f"[INFO] Loading image: {img_path}")
img = cv2.imread(img_path)

# Convert to RGB as ArcFace expects RGB input
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ---- Preprocess manually ----
# (x - 127.5) / 128.0  → ArcFace normalization
img_rgb = img_rgb.astype(np.float32)
img_rgb = (img_rgb - 127.5) / 128.0

# HWC → CHW
img_rgb = np.transpose(img_rgb, (2, 0, 1))
img_rgb = np.expand_dims(img_rgb, axis=0)  # Add batch dimension

print("[INFO] Computing embedding...")
embedding = model.forward(img_rgb)  # This gives you the 512-D feature
embedding = embedding.flatten()

# Normalize (L2)
embedding = embedding / np.linalg.norm(embedding)

print("Embedding shape:", embedding.shape)
print("Embedding norm:", np.linalg.norm(embedding))
print("First 5 dims:", embedding[:5])
