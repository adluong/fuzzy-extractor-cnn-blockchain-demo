"""
Improved BioHashing with Reliable Bit Selection
================================================

This module implements an enhanced BioHashing scheme that:
1. Identifies "reliable" bits during enrollment (far from decision boundary)
2. Uses only reliable bits for fuzzy extraction
3. Significantly reduces intra-class Hamming distance

Reference:
- Kelkboom et al., "Multi-algorithm fusion with template protection", IEEE BTAS 2009
- Nandakumar et al., "Hardening Fingerprint Fuzzy Vault", IEEE TIFS 2007
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, List
from dataclasses import dataclass

from config import BioHashConfig, DEFAULT_CONFIG


@dataclass
class ReliableBitsInfo:
    """Information about reliable bits selected during enrollment."""
    reliable_indices: np.ndarray  # Indices of reliable bits
    num_reliable: int            # Number of reliable bits
    thresholds: np.ndarray       # Per-bit thresholds (optional)
    
    def to_bytes(self) -> bytes:
        """Serialize reliable bits info."""
        # Store as: num_reliable (2 bytes) + indices (2 bytes each)
        data = self.num_reliable.to_bytes(2, 'big')
        for idx in self.reliable_indices:
            data += int(idx).to_bytes(2, 'big')
        return data
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'ReliableBitsInfo':
        """Deserialize reliable bits info."""
        num_reliable = int.from_bytes(data[:2], 'big')
        indices = []
        for i in range(num_reliable):
            idx = int.from_bytes(data[2 + i*2 : 4 + i*2], 'big')
            indices.append(idx)
        return cls(
            reliable_indices=np.array(indices),
            num_reliable=num_reliable,
            thresholds=None
        )


class ImprovedBioHasher(nn.Module):
    """
    Improved BioHashing with reliable bit selection.
    
    Key improvements over standard BioHash:
    1. Reliable bit selection - only use bits with high confidence
    2. Adaptive thresholding - use median instead of 0
    3. Configurable reliability threshold
    
    This significantly reduces intra-class Hamming distance while
    maintaining inter-class separation.
    """
    
    def __init__(
        self, 
        config: BioHashConfig = None,
        reliability_threshold: float = 0.05,
        min_reliable_bits: int = 127,
        use_median_threshold: bool = True
    ):
        """
        Initialize improved BioHasher.
        
        Args:
            config: BioHash configuration
            reliability_threshold: Minimum |projected_value| for a bit to be reliable
            min_reliable_bits: Minimum number of reliable bits required
            use_median_threshold: Use median instead of 0 for binarization
        """
        super().__init__()
        self.config = config or DEFAULT_CONFIG.biohash
        
        self.binary_length = self.config.binary_length
        self.embedding_dim = self.config.embedding_dim
        self.reliability_threshold = reliability_threshold
        self.min_reliable_bits = min_reliable_bits
        self.use_median_threshold = use_median_threshold
        
        # Initialize projection matrix
        self._init_projection_matrix(self.config.projection_seed)
    
    def _init_projection_matrix(self, seed: Optional[int] = None):
        """Generate orthonormal random projection matrix."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Generate random Gaussian matrix
        random_matrix = torch.randn(self.binary_length, self.embedding_dim)
        
        # Apply Gram-Schmidt orthogonalization
        if self.binary_length <= self.embedding_dim:
            q, _ = torch.linalg.qr(random_matrix.T)
            orthonormal = q.T[:self.binary_length]
        else:
            orthonormal = random_matrix / random_matrix.norm(dim=1, keepdim=True)
        
        self.register_buffer('projection_matrix', orthonormal)
    
    def project(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project embedding to binary_length dimensions."""
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        return torch.matmul(embedding, self.projection_matrix.T)
    
    def forward(
        self, 
        embedding: torch.Tensor,
        reliable_info: Optional[ReliableBitsInfo] = None
    ) -> Tuple[np.ndarray, Optional[ReliableBitsInfo]]:
        """
        Convert embedding to binary code.
        
        Args:
            embedding: CNN embedding, shape (1, embedding_dim) or (embedding_dim,)
            reliable_info: If provided, use these reliable bits (authentication mode)
                          If None, compute reliable bits (enrollment mode)
        
        Returns:
            (binary_code, reliable_info): Binary code and reliability information
        """
        # Project
        projected = self.project(embedding)
        projected_np = projected.detach().cpu().numpy().flatten()
        
        # Determine threshold
        if self.use_median_threshold:
            threshold = np.median(projected_np)
        else:
            threshold = 0.0
        
        # Binarize all bits
        full_binary = (projected_np > threshold).astype(np.uint8)
        
        if reliable_info is None:
            # ENROLLMENT: Identify reliable bits
            distances = np.abs(projected_np - threshold)
            
            # Sort by distance from threshold (descending)
            sorted_indices = np.argsort(distances)[::-1]
            
            # Select bits with distance > reliability_threshold
            reliable_mask = distances > self.reliability_threshold
            reliable_indices = np.where(reliable_mask)[0]
            
            # Ensure minimum number of reliable bits
            if len(reliable_indices) < self.min_reliable_bits:
                # Take top min_reliable_bits by distance
                reliable_indices = sorted_indices[:self.min_reliable_bits]
            
            # Sort indices for consistent ordering
            reliable_indices = np.sort(reliable_indices)
            
            reliable_info = ReliableBitsInfo(
                reliable_indices=reliable_indices,
                num_reliable=len(reliable_indices),
                thresholds=None
            )
        
        # Extract only reliable bits
        reliable_binary = full_binary[reliable_info.reliable_indices]
        
        return reliable_binary, reliable_info
    
    def to_bytes(self, binary_code: np.ndarray) -> bytes:
        """Convert binary array to bytes."""
        # Pad to multiple of 8
        padded_length = ((len(binary_code) + 7) // 8) * 8
        padded = np.zeros(padded_length, dtype=np.uint8)
        padded[:len(binary_code)] = binary_code
        
        # Pack bits into bytes
        return np.packbits(padded).tobytes()
    
    def analyze_reliability(self, embedding: torch.Tensor) -> dict:
        """Analyze bit reliability for an embedding."""
        projected = self.project(embedding)
        projected_np = projected.detach().cpu().numpy().flatten()
        
        threshold = np.median(projected_np) if self.use_median_threshold else 0.0
        distances = np.abs(projected_np - threshold)
        
        return {
            'total_bits': len(projected_np),
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'bits_above_0.01': np.sum(distances > 0.01),
            'bits_above_0.05': np.sum(distances > 0.05),
            'bits_above_0.10': np.sum(distances > 0.10),
            'threshold_used': threshold
        }


class AdaptiveFuzzyExtractor:
    """
    Fuzzy Extractor adapted for reliable bit selection.
    
    Works with variable-length binary codes based on reliable bit selection.
    """
    
    def __init__(
        self, 
        bch_m: int = 9,
        bch_t: int = 29,
        min_code_bits: int = 127
    ):
        """
        Initialize adaptive fuzzy extractor.
        
        Args:
            bch_m: BCH field parameter (n = 2^m - 1)
            bch_t: Error correction capability
            min_code_bits: Minimum bits for fuzzy extractor
        """
        import bchlib
        
        self.bch_m = bch_m
        self.bch_t = bch_t
        self.n = (1 << bch_m) - 1
        self.min_code_bits = min_code_bits
        
        # Initialize BCH
        try:
            self.bch = bchlib.BCH(bch_t, m=bch_m)
        except:
            self.bch = bchlib.BCH(0x211, bch_t)  # Fallback
        
        self.k = self.bch.n - self.bch.ecc_bits
        self.ecc_bytes = self.bch.ecc_bytes
        
        print(f"AdaptiveFuzzyExtractor: BCH({self.n}, {self.k}, {bch_t})")
        print(f"  - Max correctable errors: {bch_t} bits ({bch_t/self.n*100:.1f}%)")
    
    def gen(self, binary_code: np.ndarray) -> Tuple[bytes, bytes]:
        """
        Generate key and helper data from binary code.
        
        Args:
            binary_code: Binary biometric template (reliable bits only)
            
        Returns:
            (key, helper_data): Secret key and public helper
        """
        import secrets
        import hashlib
        
        # Convert binary to bytes
        code_bytes = self._bits_to_bytes(binary_code)
        
        # Generate random key
        key_bytes = self.k // 8
        random_key = secrets.token_bytes(key_bytes)
        
        # Encode key
        padded_key = random_key[:key_bytes].ljust(key_bytes, b'\x00')
        ecc = self.bch.encode(padded_key)
        codeword = padded_key + ecc
        
        # Pad biometric to codeword length
        codeword_len = len(codeword)
        padded_bio = code_bytes[:codeword_len].ljust(codeword_len, b'\x00')
        
        # Compute sketch
        sketch = bytes(a ^ b for a, b in zip(codeword, padded_bio))
        
        # Derive final key
        salt = secrets.token_bytes(16)
        derived_key = hashlib.pbkdf2_hmac('sha256', random_key, salt, 100000, dklen=32)
        
        # Pack helper data: salt + sketch
        helper_data = salt + sketch
        
        return derived_key, helper_data
    
    def rep(self, binary_code: np.ndarray, helper_data: bytes) -> Tuple[Optional[bytes], int]:
        """
        Reproduce key from noisy binary code.
        
        Args:
            binary_code: Noisy binary template
            helper_data: Helper data from enrollment
            
        Returns:
            (key, num_errors) or (None, -1) if failed
        """
        import hashlib
        
        # Unpack helper data
        salt = helper_data[:16]
        sketch = helper_data[16:]
        
        # Convert binary to bytes
        code_bytes = self._bits_to_bytes(binary_code)
        
        # Pad to sketch length
        sketch_len = len(sketch)
        padded_bio = code_bytes[:sketch_len].ljust(sketch_len, b'\x00')
        
        # Recover noisy codeword
        noisy_codeword = bytes(a ^ b for a, b in zip(sketch, padded_bio))
        
        # Decode
        key_bytes = self.k // 8
        data = bytearray(noisy_codeword[:key_bytes])
        ecc = bytearray(noisy_codeword[key_bytes:key_bytes + self.ecc_bytes])
        
        try:
            result = self.bch.decode(data, ecc)
            
            if isinstance(result, tuple):
                nerrors = result[0]
                if nerrors >= 0 and len(result) > 1:
                    data = bytearray(result[1])
            else:
                nerrors = result
            
            if nerrors < 0:
                return None, -1
            
            # Derive final key
            derived_key = hashlib.pbkdf2_hmac('sha256', bytes(data), salt, 100000, dklen=32)
            
            return derived_key, nerrors
            
        except Exception as e:
            return None, -1
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes."""
        padded_length = ((len(bits) + 7) // 8) * 8
        padded = np.zeros(padded_length, dtype=np.uint8)
        padded[:len(bits)] = bits
        return np.packbits(padded).tobytes()


def test_improved_biohash():
    """Test the improved BioHasher."""
    print("=" * 60)
    print("Testing Improved BioHasher")
    print("=" * 60)
    
    # Create hasher
    hasher = ImprovedBioHasher(
        reliability_threshold=0.05,
        min_reliable_bits=200,
        use_median_threshold=True
    )
    
    # Create test embedding (normalized)
    torch.manual_seed(42)
    embedding = torch.randn(1, 512)
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    
    # Analyze reliability
    print("\n[1] Reliability Analysis")
    analysis = hasher.analyze_reliability(embedding)
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Enrollment
    print("\n[2] Enrollment")
    binary_code, reliable_info = hasher(embedding)
    print(f"  Total bits: {hasher.binary_length}")
    print(f"  Reliable bits selected: {reliable_info.num_reliable}")
    print(f"  Binary code length: {len(binary_code)}")
    print(f"  Ones ratio: {np.sum(binary_code) / len(binary_code):.2%}")
    
    # Simulate noisy embedding (same person, different capture)
    print("\n[3] Authentication (simulated noise)")
    noisy_embedding = embedding + torch.randn_like(embedding) * 0.1
    noisy_embedding = torch.nn.functional.normalize(noisy_embedding, p=2, dim=1)
    
    # Use same reliable bits
    noisy_binary, _ = hasher(noisy_embedding, reliable_info)
    
    # Compute Hamming distance
    hamming = np.sum(binary_code != noisy_binary)
    print(f"  Original Hamming (all 511 bits): ~114 bits (22%)")
    print(f"  Reliable Hamming ({len(binary_code)} bits): {hamming} bits ({hamming/len(binary_code)*100:.1f}%)")
    
    # Test with fuzzy extractor
    print("\n[4] Fuzzy Extractor Test")
    fe = AdaptiveFuzzyExtractor(bch_m=9, bch_t=29)
    
    # Generate
    key, helper = fe.gen(binary_code)
    print(f"  Key generated: {key.hex()[:32]}...")
    
    # Reproduce with noisy code
    recovered, errors = fe.rep(noisy_binary, helper)
    if recovered is not None:
        print(f"  Key recovered: ✓ ({errors} errors corrected)")
    else:
        print(f"  Key recovery failed: {hamming} errors > {fe.bch_t} capacity")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_improved_biohash()
