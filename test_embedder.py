"""
Test embedder with actual student face images
"""

import cv2
import numpy as np
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'utils'))

from embedder import create_embedder
import glob

def test_single_image(embedder, image_path):
    """Test embedding extraction on a single image"""
    print(f"\nTesting: {os.path.basename(image_path)}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image")
        return None
    
    print(f"  Image shape: {img.shape}")
    
    try:
        embedding = embedder.get_embedding(img, normalize=True)
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  L2 norm: {np.linalg.norm(embedding):.4f}")
        print(f"  Range: [{embedding.min():.4f}, {embedding.max():.4f}]")
        return embedding
    except Exception as e:
        print(f"  Failed: {e}")
        return None

def test_similarity(embedder, img_path1, img_path2):
    """Test similarity between two images"""
    print(f"\nComparing similarity...")
    
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None or img2 is None:
        print("Failed to load images")
        return
    
    try:
        emb1 = embedder.get_embedding(img1, normalize=True)
        emb2 = embedder.get_embedding(img2, normalize=True)
        
        cos_sim = embedder.cosine_similarity(emb1, emb2)
        euc_dist = embedder.euclidean_distance(emb1, emb2)
        
        print(f"  Cosine Similarity: {cos_sim:.4f} (higher = more similar)")
        print(f"  Euclidean Distance: {euc_dist:.4f} (lower = more similar)")
        
        if cos_sim > 0.6:
            print(f"  Same person (threshold: 0.6)")
        else:
            print(f"  Different person (threshold: 0.6)")
            
    except Exception as e:
        print(f"  Failed: {e}")

def main():
    print("="*60)
    print("TESTING FACE EMBEDDER MODULE")
    print("="*60)
    
    # Initialize embedder
    embedder = create_embedder('arcface', device='cuda')
    
    # YOUR ACTUAL PATH
    aligned_dir = r"face_db\students\23CS002\aligned"
    
    if not os.path.exists(aligned_dir):
        print(f"\nDirectory not found: {aligned_dir}")
        print("Please check the path")
        return
    
    # Get all aligned images
    image_paths = glob.glob(os.path.join(aligned_dir, "*.jpg"))
    
    if len(image_paths) == 0:
        print(f"\nNo images found in {aligned_dir}")
        return
    
    print(f"\nFound {len(image_paths)} aligned images")
    
    # Test on first few images
    embeddings = []
    for i, img_path in enumerate(image_paths[:3]):
        emb = test_single_image(embedder, img_path)
        if emb is not None:
            embeddings.append(emb)
    
    # Test similarity between first two images (same person)
    if len(image_paths) >= 2:
        test_similarity(embedder, image_paths[0], image_paths[1])
    
    # Batch processing test
    print(f"\n{'='*60}")
    print("BATCH PROCESSING TEST")
    print("="*60)
    
    images = [cv2.imread(p) for p in image_paths[:5]]
    images = [img for img in images if img is not None]
    
    print(f"\nProcessing {len(images)} images in batch...")
    batch_embeddings = embedder.get_embeddings_batch(images, normalize=True)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    # Compute all pairwise similarities
    print(f"\nPairwise Cosine Similarities (same person should be >0.7):")
    for i in range(min(3, len(batch_embeddings))):
        for j in range(i+1, min(3, len(batch_embeddings))):
            sim = embedder.cosine_similarity(batch_embeddings[i], batch_embeddings[j])
            print(f"  Image {i} <-> Image {j}: {sim:.4f}")

if __name__ == "__main__":
    main()