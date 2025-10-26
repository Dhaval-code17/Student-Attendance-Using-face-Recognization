import cv2
import numpy as np
from insightface.app import FaceAnalysis

# -----------------------------
# SCRFDDetector class
# -----------------------------
class SCRFDDetector:
    def __init__(self, model_name="scrfd_10g_bnkps", model_root=r"D:\scrfd_10g_bnkps", device="cpu"):
        """
        Initialize SCRFD detector.

        Args:
            model_name: model name from InsightFace (default: scrfd_10g_bnkps)
            model_root: local folder containing SCRFD model
            device: 'cuda' for GPU, 'cpu' for CPU
        """
        providers = ["CPUExecutionProvider"]
        if device.lower() == "cuda":
            providers.insert(0, "CUDAExecutionProvider")
        self.app = FaceAnalysis(name=model_name, root=model_root, providers=providers)
        ctx_id = 0 if device.lower() == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_thresh=0.5, det_size=(640, 640))

    @staticmethod
    def align_face(img, landmark, output_size=(112, 112)):
        """
        Align face using 5-point landmarks.
        Args:
            img: BGR image
            landmark: 5-point landmarks np.array [[x1,y1],...]
            output_size: output aligned face size (default 112x112)
        Returns:
            aligned face image
        """
        ref_points = np.array([
            [38.2946, 51.6963],
            [73.5318, 51.5014],
            [56.0252, 71.7366],
            [41.5493, 92.3655],
            [70.7299, 92.2041]
        ], dtype=np.float32)
        src = landmark.astype(np.float32)
        tform = cv2.estimateAffinePartial2D(src, ref_points, method=cv2.LMEDS)[0]
        aligned = cv2.warpAffine(img, tform, output_size, borderValue=0.0)
        return aligned

    def get_aligned_faces(self, image):
        """
        Detect and align all faces in the image.

        Args:
            image: BGR image
        Returns:
            List of aligned face images
        """
        faces = self.app.get(image)
        aligned_faces = []
        for face in faces:
            aligned = self.align_face(image, face.kps)
            aligned_faces.append(aligned)
        return aligned_faces

# -----------------------------
# Optional test when run directly
# -----------------------------
if __name__ == "__main__":
    test_img = cv2.imread("face_db/students/23CS002/img_01.jpg")
    detector = SCRFDDetector()
    faces = detector.get_aligned_faces(test_img)
    for i, f in enumerate(faces):
        print(f"Face {i} detected, shape={f.shape}")
        cv2.imshow(f"Aligned_{i}", f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
