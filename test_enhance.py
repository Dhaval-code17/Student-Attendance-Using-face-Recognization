import cv2
from enhance_face import enhance_face

# Load one aligned image
img = cv2.imread("D:/SMART ATTENDANCE/face_db/students/23CS002/aligned/aligned_1.jpg")

# Enhance it
enhanced = enhance_face(img)

# Save enhanced version
cv2.imwrite("D:/SMART ATTENDANCE/face_db/students/23CS002/enhanced/test_enhanced.jpg", enhanced)

print("✅ Enhancement complete: check the 'enhanced' folder.")
