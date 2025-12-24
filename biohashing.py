"""
BioHashing: Secure Binarization of Biometric Embeddings
========================================================

This module implements BioHashing, a two-factor authentication scheme that
projects biometric features onto a user-specific random subspace before
binarization.

Key Security Properties:
1. Cancelability: If the hash is compromised, re-issue with a new random matrix
2. Irreversibility: Cannot recover the original embedding from the binary code
3. Unlinkability: Different random matrices → unlinkable binary codes

Reference: Teoh et al., "BioHashing: two factor authentication featuring 
           fingerprint data and tokenised random number", Pattern Recognition 2004.

Cryptographic Considerations:
- The random projection matrix R serves as a "token" (second factor)
- In production, R should be derived from a user-held secret (e.g., smart card)
- For integration with Fuzzy Extractors, R can be stored alongside helper data
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Union
import hashlib
import secrets

from config import BioHashConfig, DEFAULT_CONFIG


class BioHasher(nn.Module):
    """
    BioHashing layer that converts continuous embeddings to binary codes.
    
    Algorithm:
        1. Generate orthonormal random matrix R ∈ ℝ^(m × d) using Gram-Schmidt
        2. Project embedding: p = R · x
        3. Binarize: b = sign(p)
    
    The orthonormality of R ensures:
        - Maximal variance preservation (PCA-like)
        - Stable Hamming distance relationships
        
    Attributes:
        projection_matrix: Orthonormal projection matrix R
        binary_length: Output binary code length (m)
        embedding_dim: Input embedding dimension (d)
    """
    
    def __init__(self, config: BioHashConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG.biohash
        
        self.binary_length = self.config.binary_length
        self.embedding_dim = self.config.embedding_dim
        
        # Initialize with a default random matrix
        # In production, this should be user-specific
        self._init_projection_matrix(self.config.projection_seed)
        
    def _init_projection_matrix(self, seed: Optional[int] = None):
        """
        Generate orthonormal random projection matrix using Gram-Schmidt.
        
        Args:
            seed: Random seed for reproducibility (user-specific in production)
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Generate random Gaussian matrix
        random_matrix = torch.randn(self.binary_length, self.embedding_dim)
        
        # Apply Gram-Schmidt orthogonalization
        # Note: For m > d, we can only get d orthonormal vectors
        # So we typically need m ≤ d or use a different approach
        if self.binary_length <= self.embedding_dim:
            q, _ = torch.linalg.qr(random_matrix.T)
            projection = q.T[:self.binary_length]
        else:
            # For m > d: use random projection without full orthogonality
            # This is acceptable as we still get good separation
            projection = random_matrix / torch.norm(random_matrix, dim=1, keepdim=True)
        
        # Register as buffer (not a learnable parameter)
        self.register_buffer('projection_matrix', projection)
        
    def set_user_token(self, token: Union[bytes, str, int]):
        """
        Set a user-specific token to generate the projection matrix.
        
        In a two-factor system, this token is the "something you have" factor.
        
        Args:
            token: User-specific secret (e.g., from smart card, password hash)
        """
        # Derive a deterministic seed from the token
        if isinstance(token, int):
            seed = token
        else:
            if isinstance(token, str):
                token = token.encode('utf-8')
            # Use SHA-256 to derive seed
            hash_digest = hashlib.sha256(token).digest()
            seed = int.from_bytes(hash_digest[:4], byteorder='big')
        
        self._init_projection_matrix(seed)
        
    def forward(
        self, 
        embedding: torch.Tensor,
        return_continuous: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert embedding to binary code.
        
        Args:
            embedding: L2-normalized embedding vectors, shape (batch, embedding_dim)
            return_continuous: If True, also return the continuous projection
            
        Returns:
            binary_code: Binary codes as {0, 1} tensor, shape (batch, binary_length)
            continuous (optional): Pre-binarization projection values
        """
        # Project onto random subspace: p = R · x^T
        # Shape: (batch, embedding_dim) @ (embedding_dim, binary_length) = (batch, binary_length)
        continuous = torch.mm(embedding, self.projection_matrix.T)
        
        # Binarize using sign function: 1 if p > 0, else 0
        binary_code = (continuous > 0).float()
        
        if return_continuous:
            return binary_code, continuous
        return binary_code
    
    def to_bytes(self, binary_code: torch.Tensor) -> bytes:
        """
        Convert binary tensor to compact bytes representation.
        
        Args:
            binary_code: Binary tensor, shape (binary_length,) or (batch, binary_length)
            
        Returns:
            Packed bytes representation
        """
        if binary_code.dim() == 1:
            binary_code = binary_code.unsqueeze(0)
            
        result = []
        for code in binary_code:
            # Convert to numpy and pack bits
            bits = code.cpu().numpy().astype(np.uint8)
            # Pad to multiple of 8
            padded_length = ((len(bits) + 7) // 8) * 8
            padded = np.zeros(padded_length, dtype=np.uint8)
            padded[:len(bits)] = bits
            # Pack into bytes
            packed = np.packbits(padded)
            result.append(bytes(packed))
            
        return result[0] if len(result) == 1 else result
    
    def from_bytes(self, packed: bytes) -> torch.Tensor:
        """
        Convert packed bytes back to binary tensor.
        
        Args:
            packed: Bytes from to_bytes()
            
        Returns:
            Binary tensor, shape (binary_length,)
        """
        # Unpack bytes to bits
        unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
        # Trim to actual length
        binary_code = torch.tensor(unpacked[:self.binary_length], dtype=torch.float32)
        return binary_code


class AdaptiveBioHasher(BioHasher):
    """
    Enhanced BioHasher with adaptive thresholding for improved stability.
    
    Instead of using a fixed threshold (0), this version learns optimal
    thresholds per dimension to minimize intra-class Hamming distance.
    
    This is particularly important for Fuzzy Extractor integration, as
    smaller intra-class distances mean more margin for error correction.
    """
    
    def __init__(self, config: BioHashConfig = None):
        super().__init__(config)
        
        # Learnable thresholds (initialized to 0)
        self.thresholds = nn.Parameter(torch.zeros(self.binary_length))
        
    def forward(
        self, 
        embedding: torch.Tensor,
        return_continuous: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert embedding to binary code with adaptive thresholds.
        """
        continuous = torch.mm(embedding, self.projection_matrix.T)
        
        # Binarize using learned thresholds
        binary_code = (continuous > self.thresholds).float()
        
        if return_continuous:
            return binary_code, continuous
        return binary_code
    
    def compute_intra_class_loss(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute loss to minimize intra-class Hamming distance.
        
        This can be used as an auxiliary loss during training to improve
        the stability of binary codes for the same identity.
        
        Args:
            embeddings: Batch of embeddings, shape (batch, embedding_dim)
            labels: Identity labels, shape (batch,)
            
        Returns:
            Average intra-class Hamming distance (as a differentiable proxy)
        """
        continuous = torch.mm(embeddings, self.projection_matrix.T)
        
        # Use sigmoid as a differentiable approximation to step function
        soft_binary = torch.sigmoid(10.0 * (continuous - self.thresholds))
        
        # Compute pairwise distances within each class
        total_loss = 0.0
        num_pairs = 0
        
        unique_labels = labels.unique()
        for label in unique_labels:
            mask = labels == label
            class_codes = soft_binary[mask]
            
            if class_codes.size(0) < 2:
                continue
                
            # Pairwise L1 distance (approximates Hamming distance)
            n = class_codes.size(0)
            for i in range(n):
                for j in range(i + 1, n):
                    total_loss += torch.abs(class_codes[i] - class_codes[j]).sum()
                    num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / (num_pairs * self.binary_length)
        return torch.tensor(0.0, device=embeddings.device)


def compute_hamming_distance(code1: torch.Tensor, code2: torch.Tensor) -> int:
    """
    Compute Hamming distance between two binary codes.
    
    Args:
        code1, code2: Binary tensors of same shape
        
    Returns:
        Number of differing bits
    """
    return int((code1 != code2).sum().item())


def compute_hamming_distance_batch(
    codes: torch.Tensor, 
    reference: torch.Tensor
) -> torch.Tensor:
    """
    Compute Hamming distances from a batch of codes to a reference.
    
    Args:
        codes: Batch of binary codes, shape (batch, length)
        reference: Single reference code, shape (length,)
        
    Returns:
        Hamming distances, shape (batch,)
    """
    return (codes != reference.unsqueeze(0)).sum(dim=1)


def analyze_biohash_statistics(
    biohasher: BioHasher,
    embeddings: torch.Tensor,
    labels: torch.Tensor
) -> dict:
    """
    Analyze the statistical properties of BioHash codes.
    
    This is useful for:
    1. Verifying sufficient entropy for cryptographic use
    2. Measuring intra-class vs inter-class Hamming distances
    3. Estimating FAR/FRR at different thresholds
    
    Args:
        biohasher: Configured BioHasher instance
        embeddings: Batch of embeddings, shape (N, embedding_dim)
        labels: Identity labels, shape (N,)
        
    Returns:
        Dictionary of statistics
    """
    with torch.no_grad():
        codes = biohasher(embeddings)
    
    binary_length = codes.size(1)
    
    # Compute intra-class distances (same identity)
    intra_distances = []
    unique_labels = labels.unique()
    
    for label in unique_labels:
        mask = labels == label
        class_codes = codes[mask]
        n = class_codes.size(0)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = compute_hamming_distance(class_codes[i], class_codes[j])
                intra_distances.append(dist)
    
    # Compute inter-class distances (different identity)
    inter_distances = []
    # Sample to avoid O(N^2) computation
    num_samples = min(1000, len(embeddings))
    indices = np.random.choice(len(embeddings), num_samples, replace=False)
    
    for i in range(num_samples):
        for j in range(i + 1, min(i + 10, num_samples)):
            if labels[indices[i]] != labels[indices[j]]:
                dist = compute_hamming_distance(codes[indices[i]], codes[indices[j]])
                inter_distances.append(dist)
    
    intra_distances = np.array(intra_distances) if intra_distances else np.array([0])
    inter_distances = np.array(inter_distances) if inter_distances else np.array([binary_length])
    
    # Bit balance (should be ~0.5 for maximum entropy)
    bit_means = codes.mean(dim=0).cpu().numpy()
    
    return {
        "binary_length": binary_length,
        "intra_class": {
            "mean": float(np.mean(intra_distances)),
            "std": float(np.std(intra_distances)),
            "max": int(np.max(intra_distances)),
            "as_percentage": float(np.mean(intra_distances) / binary_length * 100)
        },
        "inter_class": {
            "mean": float(np.mean(inter_distances)),
            "std": float(np.std(inter_distances)),
            "min": int(np.min(inter_distances)),
            "as_percentage": float(np.mean(inter_distances) / binary_length * 100)
        },
        "bit_balance": {
            "mean": float(np.mean(bit_means)),
            "std": float(np.std(bit_means)),
            "min": float(np.min(bit_means)),
            "max": float(np.max(bit_means))
        },
        "separation_margin": float(np.mean(inter_distances) - np.max(intra_distances)),
        "estimated_entropy_bits": float(binary_length * np.mean(bit_means * np.log2(1/bit_means + 1e-10) + 
                                                                 (1-bit_means) * np.log2(1/(1-bit_means) + 1e-10)))
    }


if __name__ == "__main__":
    # Test BioHasher
    config = BioHashConfig(binary_length=511, embedding_dim=512)
    biohasher = BioHasher(config)
    
    # Simulate embeddings
    embeddings = torch.randn(10, 512)
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    # Generate binary codes
    binary_codes = biohasher(embeddings)
    print(f"Binary code shape: {binary_codes.shape}")
    print(f"Sample code (first 32 bits): {binary_codes[0, :32].int().tolist()}")
    
    # Test byte conversion
    packed = biohasher.to_bytes(binary_codes[0])
    print(f"Packed bytes length: {len(packed)} bytes")
    
    unpacked = biohasher.from_bytes(packed)
    assert torch.allclose(binary_codes[0], unpacked), "Byte conversion failed!"
    print("Byte conversion: OK")
    
    # Test Hamming distance
    dist = compute_hamming_distance(binary_codes[0], binary_codes[1])
    print(f"Hamming distance between code 0 and 1: {dist} bits ({dist/511*100:.1f}%)")
    
    # Test user token
    biohasher.set_user_token("user_secret_token_123")
    binary_codes_with_token = biohasher(embeddings)
    print(f"Codes differ with different token: {not torch.allclose(binary_codes, binary_codes_with_token)}")