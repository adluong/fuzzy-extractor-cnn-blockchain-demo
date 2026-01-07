"""
Lightweight Biometric Authentication Pipeline (No PyTorch)
==========================================================

Full pipeline using numpy-based feature extraction.
Works on systems where PyTorch crashes.

Dependencies: numpy, bchlib, cryptography (all lightweight)
"""

import numpy as np
import hashlib
import secrets
import time
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import json

# Local imports
from feature_extractor_lite import LightweightExtractor, DeterministicExtractor


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    # Feature extraction
    embedding_dim: int = 512
    
    # BioHashing
    binary_length: int = 511
    
    # Fuzzy Extractor (BCH parameters)
    bch_m: int = 9          # GF(2^9) = 512 bits
    bch_t: int = 29         # Correct up to 29 errors
    
    # Key derivation
    key_length: int = 32    # 256 bits


CONFIG = Config()


# =============================================================================
# BioHashing (numpy-only)
# =============================================================================

class BioHasher:
    """
    BioHashing: Random projection binarization.
    
    Converts continuous embeddings to binary strings using
    orthonormal random projection followed by thresholding.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        output_bits: int = 511,
        seed: int = None
    ):
        self.input_dim = input_dim
        self.output_bits = output_bits
        
        # Generate orthonormal projection matrix
        rng = np.random.RandomState(seed)
        matrix = rng.randn(input_dim, output_bits).astype(np.float32)
        
        # Gram-Schmidt orthogonalization
        self.projection, _ = np.linalg.qr(matrix)
        if self.projection.shape[1] < output_bits:
            # Pad if needed
            extra = rng.randn(input_dim, output_bits - self.projection.shape[1])
            self.projection = np.hstack([self.projection, extra])
        
        self.projection = self.projection[:, :output_bits].astype(np.float32)
    
    def hash(self, embedding: np.ndarray) -> np.ndarray:
        """
        Convert embedding to binary string.
        
        Args:
            embedding: Feature vector, shape (embedding_dim,)
            
        Returns:
            Binary array, shape (output_bits,), dtype=uint8 with values {0,1}
        """
        # Project
        projected = embedding @ self.projection
        
        # Binarize (threshold at 0)
        binary = (projected > 0).astype(np.uint8)
        
        return binary
    
    def to_bytes(self, binary: np.ndarray) -> bytes:
        """Convert binary array to bytes."""
        # Pad to multiple of 8
        padded_len = ((len(binary) + 7) // 8) * 8
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(binary)] = binary
        
        # Pack bits into bytes
        byte_array = np.packbits(padded)
        return bytes(byte_array)
    
    def from_bytes(self, data: bytes, length: int) -> np.ndarray:
        """Convert bytes back to binary array."""
        bit_array = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return bit_array[:length]


# =============================================================================
# Fuzzy Extractor (with bchlib)
# =============================================================================

class FuzzyExtractor:
    """
    Code-offset Fuzzy Extractor using BCH codes.
    
    Gen(w) -> (R, P): Generate key and helper data
    Rep(w', P) -> R:  Recover key from noisy input
    """
    
    def __init__(self, config: Config = None):
        self.config = config or CONFIG
        
        try:
            import bchlib
            # bchlib API: BCH(t, m=m) where t=error correction capability, m=GF(2^m)
            try:
                self.bch = bchlib.BCH(self.config.bch_t, m=self.config.bch_m)
            except TypeError:
                # Some versions use positional args differently
                try:
                    self.bch = bchlib.BCH(self.config.bch_m, self.config.bch_t)
                except:
                    # Try with polynomial (8219 is common for m=13)
                    self.bch = bchlib.BCH(8219, self.config.bch_t)
            
            self.n = 2**self.config.bch_m - 1  # 511
            self.k = self.n - self.bch.ecc_bits  # ~250
            self._has_bch = True
            print(f"BCH({self.n}, {self.k}, {self.config.bch_t}) initialized")
        except ImportError:
            print("WARNING: bchlib not installed. Using fallback (no error correction).")
            self._has_bch = False
            self.n = self.config.binary_length
        except Exception as e:
            print(f"WARNING: bchlib initialization failed ({e}). Using fallback.")
            self._has_bch = False
            self.n = self.config.binary_length
    
    def gen(self, w: np.ndarray) -> Tuple[bytes, bytes]:
        """
        Generate key and helper data from biometric.
        
        Args:
            w: Binary biometric, shape (n,)
            
        Returns:
            (key, helper_data): Key bytes and public helper data
        """
        if len(w) != self.n:
            raise ValueError(f"Expected {self.n} bits, got {len(w)}")
        
        if self._has_bch:
            # Generate random data that fits in BCH message capacity
            # We need to work with byte-aligned data for bchlib
            data_bytes_len = (self.n // 8) - self.bch.ecc_bytes
            random_data = secrets.token_bytes(data_bytes_len)
            ecc = self.bch.encode(random_data)
            
            # Codeword = data || ecc
            full_codeword = random_data + ecc
            codeword_bits = self._bytes_to_bits(full_codeword, len(full_codeword) * 8)
            
            # Pad codeword to n bits (add zeros)
            codeword = np.zeros(self.n, dtype=np.uint8)
            codeword[:len(codeword_bits)] = codeword_bits
            
            # Compute sketch: P = C XOR w
            sketch = np.bitwise_xor(codeword, w)
            
            # Key is hash of random data
            salt = secrets.token_bytes(16)
            key = self._derive_key(random_data, salt)
            
            # Helper data = sketch || salt || data_len (for proper decoding)
            helper_data = self._bits_to_bytes(sketch) + salt + bytes([data_bytes_len])
        else:
            # Fallback: no error correction
            salt = secrets.token_bytes(16)
            key = self._derive_key(self._bits_to_bytes(w), salt)
            helper_data = self._bits_to_bytes(w) + salt  # NOT SECURE - for testing only
        
        return key, helper_data
    
    def rep(self, w_prime: np.ndarray, helper_data: bytes) -> Optional[bytes]:
        """
        Recover key from noisy biometric and helper data.
        
        Args:
            w_prime: Noisy binary biometric, shape (n,)
            helper_data: Public helper data from Gen()
            
        Returns:
            Recovered key if successful, None if too many errors
        """
        if len(w_prime) != self.n:
            raise ValueError(f"Expected {self.n} bits, got {len(w_prime)}")
        
        if self._has_bch:
            # Parse helper data: sketch || salt || data_len
            data_bytes_len = helper_data[-1]
            salt = helper_data[-17:-1]
            sketch_bytes = helper_data[:-17]
            sketch = self._bytes_to_bits(sketch_bytes, self.n)
            
            # Compute noisy codeword: C' = P XOR w'
            noisy_codeword = np.bitwise_xor(sketch, w_prime)
            noisy_bytes = self._bits_to_bytes(noisy_codeword)
            
            # Split into data and ECC portions
            data_bytes = bytearray(noisy_bytes[:data_bytes_len])
            ecc_bytes = bytearray(noisy_bytes[data_bytes_len:data_bytes_len + self.bch.ecc_bytes])
            
            # Attempt decoding
            try:
                nerrors = self.bch.decode(data_bytes, ecc_bytes)
                if nerrors < 0:
                    return None  # Uncorrectable
                
                # Recover key
                key = self._derive_key(bytes(data_bytes), salt)
                return key
            except Exception as e:
                return None
        else:
            # Fallback
            salt = helper_data[-16:]
            stored_w = self._bytes_to_bits(helper_data[:-16], self.n)
            
            # Check Hamming distance
            errors = np.sum(w_prime != stored_w)
            if errors > self.config.bch_t:
                return None
            
            key = self._derive_key(helper_data[:-16], salt)
            return key
    
    def _derive_key(self, data: bytes, salt: bytes) -> bytes:
        """Derive key using HKDF-like construction."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=self.config.key_length,
            salt=salt,
            info=b"biometric-key-v1"
        )
        return hkdf.derive(data)
    
    def _bytes_to_bits(self, data: bytes, length: int) -> np.ndarray:
        """Convert bytes to bit array."""
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        return bits[:length]
    
    def _bits_to_bytes(self, bits: np.ndarray) -> bytes:
        """Convert bit array to bytes."""
        padded_len = ((len(bits) + 7) // 8) * 8
        padded = np.zeros(padded_len, dtype=np.uint8)
        padded[:len(bits)] = bits
        return bytes(np.packbits(padded))


# =============================================================================
# Key Derivation for Blockchain
# =============================================================================

def derive_keypair(key: bytes) -> Tuple[bytes, str]:
    """
    Derive ECDSA keypair from fuzzy extractor key.
    
    Args:
        key: 32-byte key from fuzzy extractor
        
    Returns:
        (private_key_bytes, address_string)
    """
    try:
        from eth_account import Account
        
        # Use key as private key (or derive via HKDF for extra security)
        account = Account.from_key(key)
        return key, account.address
    except ImportError:
        # Fallback without eth_account
        address = "0x" + hashlib.sha256(key).hexdigest()[:40]
        return key, address


def sign_challenge(private_key: bytes, challenge: bytes, address: str) -> bytes:
    """Sign authentication challenge."""
    try:
        from eth_account import Account
        from eth_account.messages import encode_defunct
        
        message = challenge + bytes.fromhex(address[2:])
        msg_hash = encode_defunct(message)
        signed = Account.sign_message(msg_hash, private_key)
        return signed.signature
    except ImportError:
        # Fallback signature
        return hashlib.sha256(private_key + challenge).digest()


# =============================================================================
# Complete Pipeline
# =============================================================================

class BiometricPipeline:
    """
    End-to-end biometric authentication pipeline.
    
    Enrollment: image -> embedding -> biohash -> FE.Gen() -> (key, helper_data)
    Authentication: image -> embedding -> biohash -> FE.Rep() -> key -> verify
    """
    
    def __init__(self, config: Config = None, use_deterministic: bool = False):
        self.config = config or CONFIG
        
        # Initialize components
        if use_deterministic:
            self.extractor = DeterministicExtractor(self.config.embedding_dim)
        else:
            self.extractor = LightweightExtractor(self.config.embedding_dim)
        
        self.biohasher = BioHasher(
            input_dim=self.config.embedding_dim,
            output_bits=self.config.binary_length,
            seed=12345  # Fixed for reproducibility
        )
        
        self.fuzzy_extractor = FuzzyExtractor(self.config)
        
        # Storage (in-memory, replace with blockchain in production)
        self.registered_users: Dict[str, Dict[str, Any]] = {}
    
    def enroll(self, user_id: str, image: np.ndarray) -> Dict[str, Any]:
        """
        Enroll a user with their biometric.
        
        Returns:
            Enrollment data including helper_data and address
        """
        t0 = time.time()
        
        # Extract embedding
        embedding = self.extractor.extract(image)
        
        # Binarize
        biohash = self.biohasher.hash(embedding)
        
        # Generate key and helper data
        key, helper_data = self.fuzzy_extractor.gen(biohash)
        
        # Derive keypair
        private_key, address = derive_keypair(key)
        
        # Store registration
        self.registered_users[user_id] = {
            "helper_data": helper_data,
            "address": address,
            "public_key_hash": hashlib.sha256(address.encode()).hexdigest(),
            "enrolled_at": time.time()
        }
        
        elapsed = (time.time() - t0) * 1000
        
        return {
            "user_id": user_id,
            "address": address,
            "helper_data_size": len(helper_data),
            "enrollment_time_ms": elapsed
        }
    
    def authenticate(
        self, 
        user_id: str, 
        image: np.ndarray,
        challenge: bytes = None
    ) -> Dict[str, Any]:
        """
        Authenticate a user with their biometric.
        
        Returns:
            Authentication result including success status
        """
        t0 = time.time()
        
        if user_id not in self.registered_users:
            return {"success": False, "error": "User not registered"}
        
        user = self.registered_users[user_id]
        
        # Extract embedding
        embedding = self.extractor.extract(image)
        
        # Binarize
        biohash = self.biohasher.hash(embedding)
        
        # Recover key
        key = self.fuzzy_extractor.rep(biohash, user["helper_data"])
        
        if key is None:
            elapsed = (time.time() - t0) * 1000
            return {
                "success": False,
                "error": "Key recovery failed (too many errors)",
                "auth_time_ms": elapsed
            }
        
        # Derive keypair and verify
        _, address = derive_keypair(key)
        
        if address != user["address"]:
            elapsed = (time.time() - t0) * 1000
            return {
                "success": False,
                "error": "Address mismatch",
                "auth_time_ms": elapsed
            }
        
        # Sign challenge if provided
        signature = None
        if challenge:
            private_key, _ = derive_keypair(key)
            signature = sign_challenge(private_key, challenge, address)
        
        elapsed = (time.time() - t0) * 1000
        
        return {
            "success": True,
            "address": address,
            "signature": signature.hex() if signature else None,
            "auth_time_ms": elapsed
        }
    
    def compute_hamming_distance(
        self, 
        image1: np.ndarray, 
        image2: np.ndarray
    ) -> int:
        """Compute Hamming distance between two biometric images."""
        emb1 = self.extractor.extract(image1)
        emb2 = self.extractor.extract(image2)
        
        hash1 = self.biohasher.hash(emb1)
        hash2 = self.biohasher.hash(emb2)
        
        return np.sum(hash1 != hash2)


# =============================================================================
# Demo
# =============================================================================

def simulate_biometric_noise(image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
    """Add noise to simulate biometric variation."""
    noise = np.random.randn(*image.shape).astype(np.float32) * noise_level
    noisy = image + noise
    return np.clip(noisy, 0, 1)


def run_demo():
    """Run demonstration of the biometric authentication system."""
    print("=" * 60)
    print("Biometric Authentication Demo (Lightweight Version)")
    print("=" * 60)
    
    # Initialize pipeline with deterministic extractor for reproducible demo
    pipeline = BiometricPipeline(use_deterministic=True)
    
    # Generate synthetic biometric (grayscale image)
    np.random.seed(42)
    genuine_image = np.random.rand(112, 112).astype(np.float32)
    
    # Enrollment
    print("\n[1] ENROLLMENT")
    print("-" * 40)
    result = pipeline.enroll("alice", genuine_image)
    print(f"  User: alice")
    print(f"  Address: {result['address']}")
    print(f"  Helper data size: {result['helper_data_size']} bytes")
    print(f"  Time: {result['enrollment_time_ms']:.2f} ms")
    
    # Authentication (same image)
    print("\n[2] AUTHENTICATION (same image)")
    print("-" * 40)
    result = pipeline.authenticate("alice", genuine_image)
    print(f"  Success: {result['success']}")
    print(f"  Time: {result['auth_time_ms']:.2f} ms")
    
    # Authentication (noisy image - simulates re-capture)
    print("\n[3] AUTHENTICATION (noisy image, 10% noise)")
    print("-" * 40)
    noisy_image = simulate_biometric_noise(genuine_image, noise_level=0.1)
    hamming_dist = pipeline.compute_hamming_distance(genuine_image, noisy_image)
    print(f"  Hamming distance: {hamming_dist} bits")
    result = pipeline.authenticate("alice", noisy_image)
    print(f"  Success: {result['success']}")
    print(f"  Time: {result['auth_time_ms']:.2f} ms")
    
    # Authentication (impostor)
    print("\n[4] AUTHENTICATION (impostor)")
    print("-" * 40)
    impostor_image = np.random.rand(112, 112).astype(np.float32)
    hamming_dist = pipeline.compute_hamming_distance(genuine_image, impostor_image)
    print(f"  Hamming distance: {hamming_dist} bits")
    result = pipeline.authenticate("alice", impostor_image)
    print(f"  Success: {result['success']}")
    if not result['success']:
        print(f"  Reason: {result['error']}")
    
    # Authentication with challenge
    print("\n[5] AUTHENTICATION (with challenge)")
    print("-" * 40)
    challenge = secrets.token_bytes(32)
    result = pipeline.authenticate("alice", genuine_image, challenge=challenge)
    print(f"  Success: {result['success']}")
    if result['success'] and result.get('signature'):
        print(f"  Signature: {result['signature'][:32]}...")
    elif not result['success']:
        print(f"  Reason: {result.get('error', 'Unknown')}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


def run_benchmark(num_users: int = 50, num_auth_attempts: int = 100):
    """Run performance benchmark."""
    print("=" * 60)
    print("Performance Benchmark")
    print("=" * 60)
    
    pipeline = BiometricPipeline()
    
    # Generate users
    np.random.seed(42)
    users = {}
    for i in range(num_users):
        users[f"user_{i}"] = np.random.rand(112, 112).astype(np.float32)
    
    # Enrollment benchmark
    print(f"\n[1] Enrolling {num_users} users...")
    t0 = time.time()
    for user_id, image in users.items():
        pipeline.enroll(user_id, image)
    enrollment_time = time.time() - t0
    print(f"  Total time: {enrollment_time*1000:.2f} ms")
    print(f"  Per user: {enrollment_time*1000/num_users:.2f} ms")
    
    # Genuine authentication benchmark
    print(f"\n[2] Genuine authentication ({num_auth_attempts} attempts)...")
    genuine_success = 0
    t0 = time.time()
    for i in range(num_auth_attempts):
        user_id = f"user_{i % num_users}"
        # Add small noise
        image = simulate_biometric_noise(users[user_id], noise_level=0.05)
        result = pipeline.authenticate(user_id, image)
        if result['success']:
            genuine_success += 1
    auth_time = time.time() - t0
    print(f"  Success rate: {genuine_success/num_auth_attempts*100:.1f}%")
    print(f"  Per auth: {auth_time*1000/num_auth_attempts:.2f} ms")
    
    # Impostor authentication benchmark
    print(f"\n[3] Impostor rejection ({num_auth_attempts} attempts)...")
    impostor_reject = 0
    for i in range(num_auth_attempts):
        user_id = f"user_{i % num_users}"
        # Use different user's image
        impostor_id = f"user_{(i + 1) % num_users}"
        result = pipeline.authenticate(user_id, users[impostor_id])
        if not result['success']:
            impostor_reject += 1
    print(f"  Rejection rate: {impostor_reject/num_auth_attempts*100:.1f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  False Rejection Rate (FRR): {(1-genuine_success/num_auth_attempts)*100:.2f}%")
    print(f"  False Acceptance Rate (FAR): {(1-impostor_reject/num_auth_attempts)*100:.2f}%")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "benchmark":
        run_benchmark()
    else:
        run_demo()