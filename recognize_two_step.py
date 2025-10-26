import cv2
import os
import json
from face_quality import compute_quality  # Import the compute_quality function

# --- ArcFace and FAISS initialization (dummy) ---
# Make sure these are imported and initialized as per your setup.
# import arcface_model
# from faiss import Index

# --- FAISS and student ID setup ---
student_ids = ["23CS001", "23CS002", "23CS004_John_Doe"]  # Example student IDs

# Dummy function to represent face recognition logic (for demonstration purposes)
def recognize_face(face_img, quality_score):
    # Placeholder function to perform face recognition based on the quality score
    # Here you would load your face recognition model and perform the actual matching.
    
    # Example recognition based on face quality score
    if quality_score > 0.75:
        return "Recognized Student", 0.95, "ArcFace"
    else:
        return "unknown", None, "ArcFace"

# --- Recognition and Processing ---
students_dir = r"D:\SMART ATTENDANCE\face_db\students"
for student_id in os.listdir(students_dir):
    student_path = os.path.join(students_dir, student_id, "aligned")
    if not os.path.isdir(student_path):
        continue
    
    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Compute quality score for each face
        quality_score, flags = compute_quality(img, orig_img_shape=img.shape)

        # Print debug info
        print(f"[INFO] Quality score for {img_name}: {quality_score}, Flags: {flags}")

        # Perform face recognition based on quality score
        name, confidence, method = recognize_face(img, quality_score)
        
        # Print recognition results
        print(f"[INFO] Recognition result for {img_name}: Name: {name}, Confidence: {confidence}, Method: {method}")

        # --- Save recognition results ---
        result_json_path = os.path.join(student_path, f"recognition_results_{img_name}.json")
        with open(result_json_path, "w") as f:
            json.dump({"name": name, "confidence": confidence, "method": method}, f, indent=4)
        print(f"[INFO] Saved recognition results for {img_name} → {result_json_path}")
