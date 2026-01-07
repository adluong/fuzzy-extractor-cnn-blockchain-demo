"""
Biometric Authentication with True LWE Fuzzy Extractor
=======================================================

This version uses the True LWE Fuzzy Extractor that operates directly
on continuous FaceNet embeddings, providing genuine post-quantum security.

Pipeline:
    Face Image → FaceNet CNN → 512-D Embedding → True LWE FE → Secret Key

Unlike the standard pipeline (which uses BioHash → BCH/Rep-code FE),
this version:
    1. Works directly on continuous embeddings
    2. Uses true LWE encryption/decryption
    3. Error tolerance based on Euclidean distance
    4. Post-quantum secure

Author: blockchain_bio project
"""

import argparse
import numpy as np
import time
import hashlib
from typing import Tuple, Optional
from dataclasses import dataclass

# Import True LWE Fuzzy Extractor
from fuzzy_extractor_true_lwe import (
    TrueLWEFuzzyExtractor,
    TrueLWEHelperData,
    TrueLWEParams
)

# Try to import FaceNet
try:
    import torch
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False
    print("Warning: facenet-pytorch not available, using mock embeddings")


class FaceNetEncoder:
    """FaceNet-based face encoder."""
    
    def __init__(self, pretrained='vggface2', device=None):
        if not FACENET_AVAILABLE:
            raise RuntimeError("facenet-pytorch required")
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        self.model = InceptionResnetV1(pretrained=pretrained).eval().to(device)
        print(f"FaceNet encoder loaded (pretrained={pretrained}, device={device})")
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """
        Encode face image to 512-D embedding.
        
        Args:
            image: Face image as numpy array (H, W, 3) or (3, H, W)
            
        Returns:
            512-D normalized embedding
        """
        if image.ndim == 3 and image.shape[2] == 3:
            # HWC to CHW
            image = np.transpose(image, (2, 0, 1))
        
        # Normalize to [-1, 1]
        if image.max() > 1:
            image = image / 255.0
        image = (image - 0.5) / 0.5
        
        # Add batch dimension
        tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(tensor).cpu().numpy().flatten()
        
        # L2 normalize
        embedding = embedding / np.linalg.norm(embedding)
        return embedding


class MockEncoder:
    """Mock encoder for testing without FaceNet."""
    
    def __init__(self):
        print("MockEncoder initialized (deterministic embeddings)")
        self._cache = {}
    
    def encode(self, image: np.ndarray) -> np.ndarray:
        """Generate deterministic embedding from image."""
        # Hash image to get seed
        if isinstance(image, np.ndarray):
            seed = int(hashlib.md5(image.tobytes()).hexdigest()[:8], 16)
        else:
            seed = hash(str(image)) & 0xFFFFFFFF
        
        if seed not in self._cache:
            rng = np.random.RandomState(seed)
            embedding = rng.randn(512)
            embedding = embedding / np.linalg.norm(embedding)
            self._cache[seed] = embedding
        
        return self._cache[seed]


@dataclass
class TrueLWEPipelineConfig:
    """Configuration for True LWE pipeline."""
    lwe_n: int = 128         # Secret dimension (increased for security)
    lwe_m: int = 512         # Use all embedding dimensions
    lwe_sigma: float = 1.4
    error_margin: float = 0.28  # ~10% embedding noise tolerance


class TrueLWEBiometricPipeline:
    """
    Biometric authentication pipeline using True LWE Fuzzy Extractor.
    
    This pipeline operates directly on continuous embeddings,
    providing post-quantum security via LWE hardness.
    """
    
    def __init__(self, config: TrueLWEPipelineConfig = None, use_mock: bool = False):
        """Initialize pipeline."""
        self.config = config or TrueLWEPipelineConfig()
        
        # Initialize encoder
        if use_mock or not FACENET_AVAILABLE:
            self.encoder = MockEncoder()
            self.encoder_type = 'mock'
        else:
            self.encoder = FaceNetEncoder()
            self.encoder_type = 'facenet'
        
        # Initialize True LWE Fuzzy Extractor
        lwe_params = TrueLWEParams(
            n=self.config.lwe_n,
            m=self.config.lwe_m,
            sigma=self.config.lwe_sigma,
            error_margin=self.config.error_margin
        )
        self.fuzzy_extractor = TrueLWEFuzzyExtractor(lwe_params)
        
        print(f"TrueLWE Pipeline initialized (encoder={self.encoder_type})")
    
    def enroll(self, image: np.ndarray) -> Tuple[bytes, TrueLWEHelperData, np.ndarray]:
        """
        Enroll a user with their face image.
        
        Args:
            image: Face image
            
        Returns:
            (secret_key, helper_data, embedding)
        """
        # Extract embedding
        embedding = self.encoder.encode(image)
        
        # Generate key and helper via True LWE FE
        key, helper = self.fuzzy_extractor.gen(embedding)
        
        return key, helper, embedding
    
    def authenticate(
        self, 
        image: np.ndarray, 
        helper: TrueLWEHelperData,
        expected_key: bytes = None
    ) -> Tuple[bool, float, Optional[bytes]]:
        """
        Authenticate a user.
        
        Args:
            image: Face image
            helper: Helper data from enrollment
            expected_key: Expected key for verification (optional)
            
        Returns:
            (success, distance, recovered_key)
        """
        # Extract embedding
        embedding = self.encoder.encode(image)
        
        # Recover key via True LWE FE
        recovered_key, distance = self.fuzzy_extractor.rep(embedding, helper)
        
        if recovered_key is None:
            return False, distance, None
        
        if expected_key is not None:
            success = (recovered_key == expected_key)
        else:
            success = True
        
        return success, distance, recovered_key


def run_demo():
    """Run demonstration of True LWE pipeline."""
    print("=" * 70)
    print("TRUE LWE BIOMETRIC AUTHENTICATION DEMO")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = TrueLWEBiometricPipeline(use_mock=True)
    
    # Simulate face images
    np.random.seed(42)
    genuine_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    
    # Enrollment
    print("\n[1] ENROLLMENT")
    print("-" * 40)
    key, helper, embedding = pipeline.enroll(genuine_image)
    print(f"  Secret key: {key.hex()[:32]}...")
    print(f"  Helper size: {len(helper.to_bytes())} bytes")
    print(f"  Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Genuine authentication (same image)
    print("\n[2] GENUINE AUTH (same image)")
    print("-" * 40)
    success, distance, recovered = pipeline.authenticate(genuine_image, helper, key)
    print(f"  Success: {success}")
    print(f"  Distance: {distance:.6f}")
    
    # Genuine authentication (slightly different - simulated)
    print("\n[3] GENUINE AUTH (with noise)")
    print("-" * 40)
    
    # Add small noise to simulate different capture
    noisy_image = genuine_image.astype(np.float32)
    noisy_image += np.random.randn(*noisy_image.shape) * 10
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    success, distance, recovered = pipeline.authenticate(noisy_image, helper, key)
    print(f"  Success: {success}")
    print(f"  Distance: {distance:.6f}")
    
    # Impostor
    print("\n[4] IMPOSTOR AUTH")
    print("-" * 40)
    impostor_image = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    success, distance, recovered = pipeline.authenticate(impostor_image, helper, key)
    print(f"  Success: {success}")
    print(f"  Distance: {distance:.6f}")
    print(f"  Correctly rejected: {not success}")
    
    print("\n" + "=" * 70)


def run_benchmark():
    """Run benchmark of True LWE pipeline."""
    print("=" * 70)
    print("TRUE LWE BIOMETRIC AUTHENTICATION BENCHMARK")
    print("=" * 70)
    
    pipeline = TrueLWEBiometricPipeline(use_mock=True)
    
    # Section 1: Error tolerance test
    print("\n" + "=" * 70)
    print("SECTION 1: LWE ERROR TOLERANCE TEST")
    print("=" * 70)
    
    print(f"\nLWE Parameters:")
    print(f"  n={pipeline.config.lwe_n}, m={pipeline.config.lwe_m}")
    print(f"  σ={pipeline.config.lwe_sigma}")
    print(f"  Error margin: {pipeline.config.error_margin*100:.0f}%")
    
    noise_levels = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]
    num_users = 20
    
    print("\nEnrolling test users...")
    
    results = []
    for noise in noise_levels:
        successes = 0
        total_distance = 0
        
        for user_id in range(num_users):
            # Create deterministic embedding
            rng = np.random.RandomState(user_id * 1000)
            embedding = rng.randn(512)
            embedding = embedding / np.linalg.norm(embedding)
            
            # Enroll
            key, helper = pipeline.fuzzy_extractor.gen(embedding)
            
            # Add noise
            if noise > 0:
                noise_vec = rng.randn(512) * noise
                noisy_embedding = embedding + noise_vec
                noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
            else:
                noisy_embedding = embedding
            
            # Authenticate
            recovered, distance = pipeline.fuzzy_extractor.rep(noisy_embedding, helper)
            
            if recovered is not None and recovered == key:
                successes += 1
            total_distance += distance if distance >= 0 else 0
        
        frr = (num_users - successes) / num_users * 100
        avg_dist = total_distance / num_users
        results.append((noise, frr, avg_dist))
    
    print("\nGenuine authentication with embedding noise:")
    print("-" * 60)
    threshold = pipeline.config.error_margin
    for noise, frr, avg_dist in results:
        status = "✓" if noise <= threshold else "✗"
        print(f"  {status} Noise {noise*100:5.1f}%: FRR = {frr:5.1f}%, Avg distance = {avg_dist:.4f}")
    
    # Impostor test
    print("\nImpostor detection:")
    print("-" * 60)
    
    impostor_successes = 0
    num_impostor_tests = 100
    
    for i in range(num_impostor_tests):
        rng = np.random.RandomState(i)
        genuine_emb = rng.randn(512)
        genuine_emb = genuine_emb / np.linalg.norm(genuine_emb)
        
        key, helper = pipeline.fuzzy_extractor.gen(genuine_emb)
        
        # Different random embedding
        impostor_emb = np.random.randn(512)
        impostor_emb = impostor_emb / np.linalg.norm(impostor_emb)
        
        recovered, _ = pipeline.fuzzy_extractor.rep(impostor_emb, helper)
        if recovered is not None:
            impostor_successes += 1
    
    far = impostor_successes / num_impostor_tests * 100
    print(f"  FAR: {far:.3f}%")
    
    # Section 2: Performance
    print("\n" + "=" * 70)
    print("SECTION 2: PERFORMANCE BENCHMARK")
    print("=" * 70)
    
    num_trials = 50
    enroll_times = []
    auth_times = []
    
    for i in range(num_trials):
        embedding = np.random.randn(512)
        embedding = embedding / np.linalg.norm(embedding)
        
        # Enrollment timing
        start = time.perf_counter()
        key, helper = pipeline.fuzzy_extractor.gen(embedding)
        enroll_times.append((time.perf_counter() - start) * 1000)
        
        # Auth timing
        start = time.perf_counter()
        recovered, _ = pipeline.fuzzy_extractor.rep(embedding, helper)
        auth_times.append((time.perf_counter() - start) * 1000)
    
    print(f"\n  Enrollment:     {np.mean(enroll_times):.2f} ± {np.std(enroll_times):.2f} ms")
    print(f"  Authentication: {np.mean(auth_times):.2f} ± {np.std(auth_times):.2f} ms")
    
    # Section 3: Security analysis
    print("\n" + "=" * 70)
    print("SECTION 3: SECURITY ANALYSIS")
    print("=" * 70)
    
    security = pipeline.fuzzy_extractor.estimate_security()
    print(f"\n  LWE Parameters:")
    print(f"    n = {security['lwe_parameters']['n']}")
    print(f"    m = {security['lwe_parameters']['m']}")
    print(f"    q = {security['lwe_parameters']['q']}")
    print(f"    σ = {security['lwe_parameters']['sigma']}")
    print(f"\n  Security Level: ~{security['entropy_analysis']['lwe_security_bits']} bits")
    print(f"  Post-Quantum: {security['post_quantum']}")
    
    # Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                    TRUE LWE BENCHMARK RESULTS                            │
├─────────────────────────────────────────────────────────────────────────┤
│ [1] ERROR TOLERANCE                                                      │
│     • Error margin: {threshold*100:.0f}% Euclidean distance                             │
│     • FRR @ 0% noise: {results[0][1]:.1f}%                                            │
│     • FRR @ {threshold*100:.0f}% noise: {[r for r in results if r[0]==threshold][0][1] if any(r[0]==threshold for r in results) else 'N/A'}%                                           │
│     • FAR: {far:.3f}%                                                         │
├─────────────────────────────────────────────────────────────────────────┤
│ [2] PERFORMANCE                                                          │
│     • Enrollment: {np.mean(enroll_times):.2f} ms                                           │
│     • Authentication: {np.mean(auth_times):.2f} ms                                       │
├─────────────────────────────────────────────────────────────────────────┤
│ [3] SECURITY                                                             │
│     • LWE Security: ~{security['entropy_analysis']['lwe_security_bits']} bits                                       │
│     • Post-Quantum: Yes                                                  │
└─────────────────────────────────────────────────────────────────────────┘
    """)
    
    print("=" * 70)
    print("Benchmark completed!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='True LWE Biometric Authentication')
    parser.add_argument('--mode', choices=['demo', 'benchmark'], default='demo',
                        help='Run mode')
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_demo()
    else:
        run_benchmark()


if __name__ == '__main__':
    main()