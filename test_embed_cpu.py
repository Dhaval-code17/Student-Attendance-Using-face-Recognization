import cv2
from utils.embedder_cpu import Embedder

# ✅ Use one of your actual aligned images
img_path = r"D:\SMART ATTENDANCE\face_db\students\23CS002\aligned\aligned_img_01.jpg"

print(f"[INFO] Loading image: {img_path}")
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Could not read image from {img_path}")

print("[INFO] Initializing Embedder (ArcFace - CPU)...")
e = Embedder()

print("[INFO] Getting embedding...")
emb = e.get_embedding(img)

print("\n✅ Embedding extracted successfully!")
print("Shape:", emb.shape)
print("First 5 values:", emb[:5])
