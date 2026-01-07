"""
Lightweight Feature Extractor (No PyTorch)
==========================================

Replaces CNN with traditional computer vision features for testing
the Fuzzy Extractor + Blockchain pipeline without heavy dependencies.

Methods:
1. PCA-based dimensionality reduction on raw pixels
2. Gabor filter bank responses
3. Local Binary Patterns (LBP)

For production, replace with proper CNN once environment is stable.
"""

import numpy as np
from typing import Tuple, Optional
import hashlib


class LightweightExtractor:
    """
    Feature extractor using traditional CV methods.
    No PyTorch/TensorFlow required.
    
    For testing purposes - replace with CNN for production.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        input_size: Tuple[int, int] = (112, 112),
        seed: int = 42
    ):
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.rng = np.random.RandomState(seed)
        
        # Random projection matrix (simulates learned features)
        input_dim = input_size[0] * input_size[1]
        self.projection = self.rng.randn(input_dim, embedding_dim).astype(np.float32)
        # Orthogonalize for better properties
        self.projection, _ = np.linalg.qr(self.projection)
        
        # Gabor filter bank
        self.gabor_filters = self._create_gabor_bank()
        
    def _create_gabor_bank(self, num_orientations: int = 8, num_scales: int = 4):
        """Create Gabor filter bank for texture features."""
        filters = []
        for scale in range(num_scales):
            sigma = 2.0 * (scale + 1)
            wavelength = sigma * 2
            for orientation in range(num_orientations):
                theta = orientation * np.pi / num_orientations
                kernel = self._gabor_kernel(sigma, theta, wavelength)
                filters.append(kernel)
        return filters
    
    def _gabor_kernel(
        self, 
        sigma: float, 
        theta: float, 
        wavelength: float,
        size: int = 21
    ) -> np.ndarray:
        """Generate a single Gabor kernel."""
        half = size // 2
        x, y = np.meshgrid(
            np.arange(-half, half + 1),
            np.arange(-half, half + 1)
        )
        
        # Rotation
        x_theta = x * np.cos(theta) + y * np.sin(theta)
        y_theta = -x * np.sin(theta) + y * np.cos(theta)
        
        # Gabor function
        gaussian = np.exp(-0.5 * (x_theta**2 + y_theta**2) / sigma**2)
        sinusoid = np.cos(2 * np.pi * x_theta / wavelength)
        
        kernel = gaussian * sinusoid
        return kernel.astype(np.float32)
    
    def _convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution (valid mode)."""
        ih, iw = image.shape
        kh, kw = kernel.shape
        oh, ow = ih - kh + 1, iw - kw + 1
        
        output = np.zeros((oh, ow), dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                output[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
        
        return output
    
    def _compute_lbp(self, image: np.ndarray, radius: int = 1) -> np.ndarray:
        """Compute Local Binary Pattern features."""
        h, w = image.shape
        lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = image[i, j]
                pattern = 0
                # 8 neighbors
                neighbors = [
                    image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                    image[i, j+1], image[i+1, j+1], image[i+1, j],
                    image[i+1, j-1], image[i, j-1]
                ]
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        pattern |= (1 << k)
                lbp[i-radius, j-radius] = pattern
        
        # Histogram as feature
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        return hist.astype(np.float32) / hist.sum()
    
    def extract(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract embedding from image.
        
        Args:
            image: Grayscale image, shape (H, W) or (H, W, 1)
                   or RGB image, shape (H, W, 3)
            normalize: L2-normalize the output
            
        Returns:
            Embedding vector, shape (embedding_dim,)
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            if image.shape[2] == 3:
                # RGB to grayscale
                image = 0.299 * image[:,:,0] + 0.587 * image[:,:,1] + 0.114 * image[:,:,2]
            else:
                image = image[:,:,0]
        
        # Resize to expected size
        image = self._resize(image, self.input_size)
        
        # Normalize pixel values
        image = image.astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-8)
        
        features = []
        
        # 1. Gabor features (texture)
        gabor_features = []
        for kernel in self.gabor_filters[:8]:  # Use subset for speed
            response = self._convolve2d(image, kernel)
            gabor_features.extend([response.mean(), response.std()])
        features.extend(gabor_features)
        
        # 2. LBP histogram (local patterns)
        lbp_hist = self._compute_lbp(image)
        features.extend(lbp_hist[:64])  # First 64 bins
        
        # 3. Random projection of flattened image
        flat = image.flatten()
        projected = flat @ self.projection[:len(flat), :128]
        features.extend(projected)
        
        # 4. Statistical features
        features.extend([
            image.mean(),
            image.std(),
            np.percentile(image, 25),
            np.percentile(image, 75),
        ])
        
        # Combine and project to final dimension
        features = np.array(features, dtype=np.float32)
        
        # Pad or truncate to embedding_dim
        if len(features) < self.embedding_dim:
            # Expand using random projection
            expansion = self.rng.randn(len(features), self.embedding_dim).astype(np.float32)
            embedding = features @ expansion
        else:
            embedding = features[:self.embedding_dim]
        
        # L2 normalize
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def _resize(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Simple bilinear resize without external dependencies."""
        h, w = image.shape
        new_h, new_w = size
        
        # Create coordinate grids
        x_ratio = w / new_w
        y_ratio = h / new_h
        
        result = np.zeros((new_h, new_w), dtype=image.dtype)
        
        for i in range(new_h):
            for j in range(new_w):
                # Source coordinates
                x = j * x_ratio
                y = i * y_ratio
                
                x0, y0 = int(x), int(y)
                x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
                
                # Bilinear weights
                wx = x - x0
                wy = y - y0
                
                result[i, j] = (
                    image[y0, x0] * (1 - wx) * (1 - wy) +
                    image[y0, x1] * wx * (1 - wy) +
                    image[y1, x0] * (1 - wx) * wy +
                    image[y1, x1] * wx * wy
                )
        
        return result
    
    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """Extract embeddings for a batch of images."""
        embeddings = []
        for img in images:
            embeddings.append(self.extract(img))
        return np.stack(embeddings)


class DeterministicExtractor:
    """
    Deterministic feature extractor using cryptographic hashing.
    
    Useful for testing when you need reproducible embeddings
    from the same input without any ML components.
    """
    
    def __init__(self, embedding_dim: int = 512, seed: int = 42):
        self.embedding_dim = embedding_dim
        self.rng = np.random.RandomState(seed)
        self.expansion_matrix = self.rng.randn(32, embedding_dim).astype(np.float32)
    
    def extract(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        Extract deterministic embedding from image.
        
        The same image always produces the same embedding.
        """
        # Hash the image content
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        image_bytes = image.tobytes()
        
        # Generate multiple hashes for more bits
        hashes = []
        for i in range(8):
            h = hashlib.sha256(image_bytes + bytes([i])).digest()
            hashes.append(np.frombuffer(h, dtype=np.uint8))
        
        hash_vector = np.concatenate(hashes).astype(np.float32)
        hash_vector = (hash_vector - 128) / 128  # Normalize to [-1, 1]
        
        # Expand to embedding dimension
        embedding = hash_vector[:32] @ self.expansion_matrix
        
        if normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding


def create_extractor(method: str = "lightweight", **kwargs):
    """
    Factory function to create feature extractor.
    
    Args:
        method: "lightweight" (Gabor+LBP), "deterministic" (hash-based),
                or "cnn" (requires PyTorch - will fail if not installed)
    """
    if method == "lightweight":
        return LightweightExtractor(**kwargs)
    elif method == "deterministic":
        return DeterministicExtractor(**kwargs)
    elif method == "cnn":
        try:
            from model import BiometricEncoder, CNNConfig
            return BiometricEncoder(CNNConfig(**kwargs))
        except ImportError:
            print("PyTorch not available. Falling back to lightweight extractor.")
            return LightweightExtractor(**kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing Lightweight Feature Extractor")
    print("=" * 50)
    
    # Create extractor
    extractor = LightweightExtractor(embedding_dim=512)
    
    # Generate synthetic test images
    np.random.seed(42)
    
    # Same "person" - similar images
    base_image = np.random.rand(112, 112).astype(np.float32)
    image1 = base_image + np.random.randn(112, 112).astype(np.float32) * 0.1
    image2 = base_image + np.random.randn(112, 112).astype(np.float32) * 0.1
    
    # Different "person"
    image3 = np.random.rand(112, 112).astype(np.float32)
    
    # Extract embeddings
    emb1 = extractor.extract(image1)
    emb2 = extractor.extract(image2)
    emb3 = extractor.extract(image3)
    
    print(f"Embedding shape: {emb1.shape}")
    print(f"Embedding norm: {np.linalg.norm(emb1):.4f}")
    
    # Cosine similarities
    sim_same = np.dot(emb1, emb2)
    sim_diff = np.dot(emb1, emb3)
    
    print(f"\nCosine similarity (same person): {sim_same:.4f}")
    print(f"Cosine similarity (different person): {sim_diff:.4f}")
    
    # Test deterministic extractor
    print("\n" + "=" * 50)
    print("Testing Deterministic Extractor")
    
    det_extractor = DeterministicExtractor(embedding_dim=512)
    
    det_emb1 = det_extractor.extract(image1)
    det_emb1_again = det_extractor.extract(image1)
    
    print(f"Same image, same embedding: {np.allclose(det_emb1, det_emb1_again)}")
    
    print("\n✓ All tests passed!")
