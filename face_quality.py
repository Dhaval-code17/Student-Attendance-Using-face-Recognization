import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# ---- Metric functions ----
def blur_score(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def exposure_score(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) / 255.0

def face_size_score(face_img, orig_img_shape):
    face_area = face_img.shape[0] * face_img.shape[1]
    total_area = orig_img_shape[0] * orig_img_shape[1]
    return face_area / total_area

def compute_quality(face_img, orig_img_shape=None,
                    blur_thresh=100.0, size_thresh=0.02,
                    exposure_low=0.3, exposure_high=0.7):
    blur = blur_score(face_img)
    exposure = exposure_score(face_img)
    size = face_size_score(face_img, orig_img_shape) if orig_img_shape is not None else 1.0

    blur_norm = min(blur / blur_thresh, 1.0)
    size_norm = min(size / size_thresh, 1.0)
    exposure_norm = np.clip((exposure - exposure_low) / (exposure_high - exposure_low), 0, 1)

    quality_score = 0.4*blur_norm + 0.3*size_norm + 0.3*exposure_norm

    flags = {
        'blurred': blur < blur_thresh,
        'small': size < size_thresh,
        'underexposed': exposure < exposure_low,
        'overexposed': exposure > exposure_high
    }
    flags = {k: bool(v) for k, v in flags.items()}

    return float(quality_score), flags

# ---- Dataset processing ----
students_dir = r"D:\SMART ATTENDANCE\face_db\students"

for student_id in os.listdir(students_dir):
    student_path = os.path.join(students_dir, student_id, "aligned")
    if not os.path.isdir(student_path):
        continue

    results = {}
    blur_flags, small_flags, underexp_flags, overexp_flags = [], [], [], []

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        score, flags = compute_quality(img, orig_img_shape=img.shape)
        results[img_name] = {'score': score, 'flags': flags}

        blur_flags.append(flags['blurred'])
        small_flags.append(flags['small'])
        underexp_flags.append(flags['underexposed'])
        overexp_flags.append(flags['overexposed'])

    # ---- Save JSON per student ----
    json_path = os.path.join(students_dir, student_id, "face_quality_scores.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Saved JSON for {student_id} → {json_path}")

    # ---- Optional histogram per student ----
    plt.figure(figsize=(12,5))
    plt.subplot(1,4,1); plt.hist(blur_flags, bins=2); plt.title('Blurred')
    plt.subplot(1,4,2); plt.hist(small_flags, bins=2); plt.title('Small')
    plt.subplot(1,4,3); plt.hist(underexp_flags, bins=2); plt.title('Underexposed')
    plt.subplot(1,4,4); plt.hist(overexp_flags, bins=2); plt.title('Overexposed')
    hist_path = os.path.join(students_dir, student_id, "face_quality_hist.png")
    plt.savefig(hist_path)
    plt.close()
    print(f"[INFO] Saved histogram for {student_id} → {hist_path}")
