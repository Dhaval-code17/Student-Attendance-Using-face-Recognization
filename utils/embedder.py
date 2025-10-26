# utils/embedder_cpu.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis

class Embedder:
    def __init__(self, model_name="buffalo_l", device="cpu"):
        ctx_id = 0 if device == "cuda" else -1
        # load detection + recognition
        self.app = FaceAnalysis(name=model_name)
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def get_embedding(self, img):
        """Return 512-D embedding for the first detected face in a BGR image."""
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected in the image")
        emb = faces[0].normed_embedding
        return emb.astype(np.float32)

    def get_embeddings(self, img_list):
        embs = []
        for img in img_list:
            emb = self.get_embedding(img)
            embs.append(emb)
        return np.stack(embs)

    @staticmethod
    def cosine_similarity(emb1, emb2):
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
