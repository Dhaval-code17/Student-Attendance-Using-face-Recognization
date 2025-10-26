import cv2
from gfpgan import GFPGANer

# Initialize GFPGAN once
gfpganer = GFPGANer(
    model_path='GFPGAN/experiments/pretrained_models/GFPGANv1.3.pth',
    upscale=1,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None
)

def enhance_face(face_img):
    """Enhance a single aligned face image (numpy array, BGR)."""
    cropped_faces, restored_faces, _ = gfpganer.enhance(
        face_img, has_aligned=True, only_center_face=True, paste_back=False
    )
    return restored_faces[0]
