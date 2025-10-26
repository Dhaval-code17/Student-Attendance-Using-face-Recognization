"""
Feature Extraction Module for SMART ATTENDANCE
Supports ArcFace (InsightFace) and AdaFace
"""

import numpy as np
import cv2
import insightface
from insightface.app import FaceAnalysis
import torch
from typing import Union, List
import os

class FaceEmbedder:
    """
    Unified face embedding extractor
    Supports: ArcFace (via InsightFace) and AdaFace
    """
    
    def __init__(self, model_type='arcface', device='cuda'):
        """
        Args:
            model_type: 'arcface' or 'adaface'
            device: 'cuda' or 'cpu'
        """
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        print(f"Initializing {model_type} on {self.device}...")
        
        if model_type == 'arcface':
            self._init_arcface()
        elif model_type == 'adaface':
            self._init_adaface()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    def _init_arcface(self):
        """Initialize InsightFace ArcFace model"""
        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=0 if self.device == 'cuda' else -1, det_size=(640, 640))
        print("ArcFace (buffalo_l) loaded successfully")
    
    def _init_adaface(self):
        """Initialize AdaFace model (optional)"""
        try:
            print("AdaFace not yet implemented - use 'arcface' for now")
            raise NotImplementedError("AdaFace support coming soon")
        except Exception as e:
            print(f"AdaFace initialization failed: {e}")
            raise
    
    def get_embedding(self, image: np.ndarray, normalize=True) -> np.ndarray:
        """
        Extract 512-D embedding from a face image
        
        Args:
            image: BGR image (OpenCV format) or RGB
            normalize: L2-normalize the embedding (recommended for cosine similarity)
        
        Returns:
            512-D numpy array
        """
        if self.model_type == 'arcface':
            return self._get_arcface_embedding(image, normalize)
        elif self.model_type == 'adaface':
            return self._get_adaface_embedding(image, normalize)
    
    def _get_arcface_embedding(self, image: np.ndarray, normalize=True) -> np.ndarray:
        """Extract embedding using ArcFace"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        faces = self.app.get(image)
        
        if len(faces) == 0:
            raise ValueError("No face detected in image")
        
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        
        embedding = face.embedding
        
        if normalize:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.astype(np.float32)
    
    def _get_adaface_embedding(self, image: np.ndarray, normalize=True) -> np.ndarray:
        """Extract embedding using AdaFace (to be implemented)"""
        raise NotImplementedError("AdaFace support coming soon")
    
    def get_embeddings_batch(self, images: List[np.ndarray], normalize=True) -> np.ndarray:
        """
        Extract embeddings from multiple images
        
        Args:
            images: List of BGR images
            normalize: L2-normalize embeddings
        
        Returns:
            (N, 512) numpy array
        """
        embeddings = []
        for img in images:
            try:
                emb = self.get_embedding(img, normalize=normalize)
                embeddings.append(emb)
            except Exception as e:
                print(f"Failed to extract embedding: {e}")
                embeddings.append(np.zeros(512, dtype=np.float32))
        
        return np.array(embeddings)
    
    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            emb1, emb2: Normalized embeddings
        
        Returns:
            Similarity score in [-1, 1] (higher = more similar)
        """
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    @staticmethod
    def euclidean_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute Euclidean distance between two embeddings
        
        Returns:
            Distance (lower = more similar)
        """
        return np.linalg.norm(emb1 - emb2)


def create_embedder(model_type='arcface', device='cuda'):
    """Factory function to create embedder"""
    return FaceEmbedder(model_type=model_type, device=device)


if __name__ == "__main__":
    print("Testing FaceEmbedder...")
    embedder = create_embedder('arcface')
    
    test_img = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    print(f"Test image shape: {test_img.shape}")
    
    try:
        embedding = embedder.get_embedding(test_img)
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        print(f"Embedding range: [{embedding.min():.4f}, {embedding.max():.4f}]")
    except Exception as e:
        print(f"Test with random image failed (expected): {e}")
        print("This is normal - random images don't contain faces")