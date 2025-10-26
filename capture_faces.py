import cv2
import os
import argparse
from insightface.app import FaceAnalysis

# -------------------- Arguments --------------------
parser = argparse.ArgumentParser(description="Capture student face images")
parser.add_argument("--id", type=str, required=True, help="Student ID")
parser.add_argument("--name", type=str, required=True, help="Student Name")
parser.add_argument("--num", type=int, default=10, help="Number of images to capture")
args = parser.parse_args()

# -------------------- Prepare directories --------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # folder of the script
STUDENTS_DIR = os.path.join(BASE_DIR, "face_db", "students")
student_folder = os.path.join(STUDENTS_DIR, f"{args.id}_{args.name.replace(' ', '_')}")
os.makedirs(student_folder, exist_ok=True)

print(f"[INFO] Images will be saved to: {student_folder}")

# -------------------- Initialize Face Detector --------------------
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU mode

# -------------------- Start Webcam --------------------
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"[INFO] Starting capture for {args.name} ({args.id})")
count = 0

while count < args.num:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to capture frame from webcam.")
        break

    faces = app.get(frame)

    if len(faces) > 0:
        # Crop first detected face
        x1, y1, x2, y2 = faces[0].bbox.astype(int)
        face_crop = frame[y1:y2, x1:x2]

        img_path = os.path.join(student_folder, f"img_{count+1:02d}.jpg")
        cv2.imwrite(img_path, face_crop)
        print(f"[INFO] Saved {img_path}")
        count += 1

    # Draw rectangles for feedback
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show live feed
    cv2.imshow("Capture Faces - Press 'q' to quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Capture complete.")
