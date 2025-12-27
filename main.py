"""
End-to-End Biometric Authentication System
============================================

This module demonstrates the complete pipeline:
    1. CNN Feature Extraction (with simulated training)
    2. BioHashing for Binarization
    3. Fuzzy Extractor for Key Generation
    4. Blockchain Registration and Authentication

Usage:
    python main.py --mode demo       # Run end-to-end demonstration
    python main.py --mode train      # Train CNN on real dataset
    python main.py --mode benchmark  # Run security benchmarks
"""

import argparse
import os
import sys
import time
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Local imports
from config import SystemConfig, DEFAULT_CONFIG
from model import BiometricModel, BiometricEncoder, create_model
from biohashing import BioHasher, analyze_biohash_statistics
from fuzzy_extractor import FuzzyExtractor, HelperData
from blockchain_client import (
    BiometricAuthClient, 
    MockBlockchainClient, 
    derive_keypair,
    BiometricKeyPair
)

# Check if FaceNet is available (in model.py)
try:
    from model import FaceNetEncoder, HAS_FACENET
except ImportError:
    HAS_FACENET = False
    FaceNetEncoder = None


class BiometricPipeline:
    """
    Complete biometric authentication pipeline.
    
    This class orchestrates all components:
        CNN → BioHasher → FuzzyExtractor → Blockchain
    
    Attributes:
        encoder: Trained CNN for feature extraction
        biohasher: BioHashing layer for binarization
        fuzzy_extractor: Cryptographic key generation
        blockchain: Blockchain client for decentralized auth
    """
    
    def __init__(
        self, 
        config: SystemConfig = None,
        use_mock_blockchain: bool = True,
        use_facenet: bool = True
    ):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        print("Initializing Biometric Pipeline...")
        
        # 1. CNN Encoder - prefer FaceNet for production-quality embeddings
        print("  [1/4] Loading CNN encoder...")
        if use_facenet and HAS_FACENET:
            try:
                self.encoder = FaceNetEncoder(pretrained='vggface2')
                self._encoder_type = 'facenet'
            except Exception as e:
                print(f"    FaceNet init failed: {e}")
                print("    Falling back to default encoder...")
                self.encoder = BiometricEncoder(self.config.cnn)
                self.encoder.eval()
                self._encoder_type = 'resnet'
        else:
            self.encoder = BiometricEncoder(self.config.cnn)
            self.encoder.eval()
            self._encoder_type = 'resnet'
            if use_facenet and not HAS_FACENET:
                print("    WARNING: FaceNet requested but not installed")
                print("    Install with: pip install facenet-pytorch")
        
        print(f"    Encoder type: {self._encoder_type}")
        
        # 2. BioHasher
        print("  [2/4] Initializing BioHasher...")
        self.biohasher = BioHasher(self.config.biohash)
        
        # 3. Fuzzy Extractor
        print("  [3/4] Initializing Fuzzy Extractor...")
        self.fuzzy_extractor = FuzzyExtractor(self.config.fuzzy_extractor)
        
        # 4. Blockchain Client
        print("  [4/4] Initializing Blockchain Client...")
        if use_mock_blockchain:
            self.blockchain = MockBlockchainClient()
        else:
            self.blockchain = BiometricAuthClient(self.config.blockchain)
        self.blockchain.connect()
        
        print("Pipeline initialized successfully!\n")
    
    @property
    def input_size(self) -> Tuple[int, int]:
        """Get the expected input image size for the encoder."""
        if hasattr(self.encoder, 'input_size'):
            return self.encoder.input_size
        elif self._encoder_type == 'facenet':
            return (160, 160)
        else:
            return (112, 112)
    
    def extract_features(self, image: torch.Tensor) -> torch.Tensor:
        """
        Extract normalized embedding from biometric image.
        
        Args:
            image: Input image tensor, shape (1, 3, H, W) or (3, H, W)
            
        Returns:
            L2-normalized embedding, shape (1, embedding_dim)
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        with torch.no_grad():
            if self._encoder_type == 'facenet':
                # FaceNet handles preprocessing internally
                embedding = self.encoder(image)
            else:
                # Original ResNet encoder
                embedding = self.encoder(image)
        
        # Ensure embedding is 2D: (1, embedding_dim)
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        return embedding
    
    def binarize(
        self, 
        embedding: torch.Tensor,
        user_token: Optional[bytes] = None
    ) -> bytes:
        """
        Convert embedding to binary code using BioHashing.
        
        Args:
            embedding: CNN embedding, shape (1, embedding_dim)
            user_token: Optional user-specific token for two-factor auth
            
        Returns:
            Binary code as bytes
        """
        if user_token:
            self.biohasher.set_user_token(user_token)
        
        binary_code = self.biohasher(embedding)
        return self.biohasher.to_bytes(binary_code[0])
    
    def enroll(
        self, 
        image: torch.Tensor,
        user_token: Optional[bytes] = None
    ) -> Tuple[bytes, HelperData, BiometricKeyPair]:
        """
        Enroll a user's biometric.
        
        Complete enrollment pipeline:
            Image → Embedding → Binary Code → (Key, HelperData) → Keypair
        
        Args:
            image: Biometric image
            user_token: Optional user-specific token
            
        Returns:
            (secret_key, helper_data, keypair)
        """
        # Extract features
        embedding = self.extract_features(image)
        
        # Binarize
        binary_code = self.binarize(embedding, user_token)
        
        # Generate key
        secret_key, helper_data = self.fuzzy_extractor.gen(binary_code)
        
        # Derive keypair
        keypair = derive_keypair(secret_key)
        
        return secret_key, helper_data, keypair
    
    def register_on_chain(
        self, 
        image: torch.Tensor,
        user_token: Optional[bytes] = None
    ) -> Tuple[str, HelperData, BiometricKeyPair]:
        """
        Complete on-chain registration.
        
        Args:
            image: Biometric image
            user_token: Optional user-specific token
            
        Returns:
            (tx_hash, helper_data, keypair)
        """
        # Enroll
        secret_key, helper_data, keypair = self.enroll(image, user_token)
        
        # Register on blockchain
        tx_hash = self.blockchain.register(
            helper_data.to_bytes(),
            keypair
        )
        
        return tx_hash, helper_data, keypair
    
    def authenticate(
        self,
        image: torch.Tensor,
        helper_data: HelperData,
        user_token: Optional[bytes] = None
    ) -> Tuple[bool, int]:
        """
        Authenticate a user.
        
        Args:
            image: Fresh biometric capture
            helper_data: Stored helper data from enrollment
            user_token: Optional user-specific token
            
        Returns:
            (success, num_errors_corrected)
        """
        # Extract features
        embedding = self.extract_features(image)
        
        # Binarize
        binary_code = self.binarize(embedding, user_token)
        
        # Recover key
        recovered_key, num_errors = self.fuzzy_extractor.rep(binary_code, helper_data)
        
        if recovered_key is None:
            return False, -1
        
        # Derive keypair and authenticate
        keypair = derive_keypair(recovered_key)
        success = self.blockchain.authenticate(keypair)
        
        return success, num_errors
    
    def authenticate_on_chain(
        self,
        image: torch.Tensor,
        user_address: str,
        user_token: Optional[bytes] = None
    ) -> Tuple[bool, int]:
        """
        Complete on-chain authentication.
        
        Fetches helper data from blockchain and authenticates.
        
        Args:
            image: Fresh biometric capture
            user_address: Ethereum address of the user
            user_token: Optional user-specific token
            
        Returns:
            (success, num_errors_corrected)
        """
        # Fetch helper data from chain
        helper_bytes = self.blockchain.get_helper_data(user_address)
        helper_data = HelperData.from_bytes(helper_bytes)
        
        return self.authenticate(image, helper_data, user_token)


def simulate_biometric_noise(
    embedding: torch.Tensor,
    noise_level: float = 0.1
) -> torch.Tensor:
    """
    Simulate noise in biometric capture.
    
    This models real-world variations due to:
    - Different lighting conditions
    - Pose variations
    - Sensor noise
    - Partial occlusion
    
    Args:
        embedding: Original embedding
        noise_level: Standard deviation of Gaussian noise
        
    Returns:
        Noisy embedding (re-normalized)
    """
    noise = torch.randn_like(embedding) * noise_level
    noisy = embedding + noise
    return torch.nn.functional.normalize(noisy, p=2, dim=1)


def run_demo():
    """
    Run end-to-end demonstration of the system.
    """
    print("=" * 70)
    print("CNN + FUZZY EXTRACTOR + BLOCKCHAIN AUTHENTICATION DEMO")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = BiometricPipeline(use_mock_blockchain=True)
    
    # Simulate biometric image (in real system, this would be a face/fingerprint)
    print("\n[ENROLLMENT]")
    print("-" * 40)
    
    # Generate a synthetic "biometric" (random image that represents the user)
    # Use the correct input size for the encoder
    input_size = pipeline.input_size
    torch.manual_seed(42)  # For reproducibility
    original_image = torch.randn(1, 3, input_size[0], input_size[1])
    print(f"Captured biometric image: {original_image.shape}")
    
    # Optional: User token for two-factor authentication
    user_token = b"user_secret_token_123"
    
    # Enrollment
    print("\nProcessing enrollment...")
    start_time = time.time()
    secret_key, helper_data, keypair = pipeline.enroll(original_image, user_token)
    enrollment_time = time.time() - start_time
    
    print(f"  Secret key (first 16 bytes): {secret_key.hex()[:32]}...")
    print(f"  Helper data size: {len(helper_data.to_bytes())} bytes")
    print(f"  Derived address: {keypair.address}")
    print(f"  Enrollment time: {enrollment_time*1000:.2f} ms")
    
    # Simulate blockchain registration
    print("\nRegistering on blockchain...")
    tx_hash = pipeline.blockchain.register(helper_data.to_bytes(), keypair)
    print(f"  Transaction: {tx_hash[:16]}...")
    
    # =========================================================================
    # AUTHENTICATION - Same biometric (should succeed)
    # =========================================================================
    print("\n[AUTHENTICATION - Same Biometric]")
    print("-" * 40)
    
    start_time = time.time()
    success, errors = pipeline.authenticate(original_image, helper_data, user_token)
    auth_time = time.time() - start_time
    
    print(f"  Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"  Errors corrected: {errors}")
    print(f"  Authentication time: {auth_time*1000:.2f} ms")
    
    # =========================================================================
    # AUTHENTICATION - Noisy biometric (should succeed with error correction)
    # =========================================================================
    print("\n[AUTHENTICATION - Noisy Biometric (5% noise)]")
    print("-" * 40)
    
    # Get embedding and add noise
    embedding = pipeline.extract_features(original_image)
    noisy_embedding = simulate_biometric_noise(embedding, noise_level=0.1)
    
    # Create a wrapper that returns the noisy embedding
    original_forward = pipeline.encoder.forward
    pipeline.encoder.forward = lambda x, **kwargs: noisy_embedding
    
    start_time = time.time()
    success, errors = pipeline.authenticate(original_image, helper_data, user_token)
    auth_time = time.time() - start_time
    
    # Restore encoder
    pipeline.encoder.forward = original_forward
    
    print(f"  Result: {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"  Errors corrected: {errors}")
    print(f"  Authentication time: {auth_time*1000:.2f} ms")
    
    # =========================================================================
    # AUTHENTICATION - Different user (should fail)
    # =========================================================================
    print("\n[AUTHENTICATION - Different User (Impostor)]")
    print("-" * 40)
    
    # Generate a different "user's" biometric
    impostor_image = torch.randn(1, 3, input_size[0], input_size[1])
    
    start_time = time.time()
    success, errors = pipeline.authenticate(impostor_image, helper_data, user_token)
    auth_time = time.time() - start_time
    
    print(f"  Result: {'✗ REJECTED (correct)' if not success else '✓ ACCEPTED (security breach!)'}")
    print(f"  Errors detected: {errors if errors >= 0 else 'Uncorrectable'}")
    print(f"  Authentication time: {auth_time*1000:.2f} ms")
    
    # =========================================================================
    # SECURITY ANALYSIS
    # =========================================================================
    print("\n[SECURITY ANALYSIS]")
    print("-" * 40)
    
    security = pipeline.fuzzy_extractor.estimate_security(biometric_entropy=200)
    
    print(f"  BCH Parameters: ({security['code_parameters']['n']}, "
          f"{security['code_parameters']['k']}, {security['code_parameters']['t']})")
    print(f"  Error tolerance: {security['error_tolerance']['max_error_rate']*100:.1f}% "
          f"({security['error_tolerance']['max_errors']} bits)")
    print(f"  Entropy leakage: {security['entropy_analysis']['leakage_bits']} bits")
    print(f"  Effective security: {security['entropy_analysis']['effective_security_bits']:.0f} bits")
    print(f"  Security level: {security['recommendation']}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)


def flip_bits(binary_bytes: bytes, num_flips: int) -> bytes:
    """Flip exactly num_flips random bits in binary code."""
    code = bytearray(binary_bytes)
    num_bits = len(code) * 8
    
    if num_flips > num_bits:
        num_flips = num_bits
    
    flip_positions = np.random.choice(num_bits, num_flips, replace=False)
    
    for pos in flip_positions:
        byte_idx = pos // 8
        bit_idx = pos % 8
        code[byte_idx] ^= (1 << bit_idx)
    
    return bytes(code)


def run_benchmark():
    """
    Run comprehensive security benchmarks including:
    1. BCH Unit Test (binary-level noise)
    2. Standard BioHash evaluation
    3. Improved BioHash with reliable bit selection
    4. Blockchain gas cost evaluation
    """
    print("=" * 75)
    print("COMPREHENSIVE BIOMETRIC AUTHENTICATION BENCHMARK")
    print("=" * 75)
    
    # =========================================================================
    # SECTION 1: BCH UNIT TEST (Binary-Level Noise)
    # =========================================================================
    print("\n" + "=" * 75)
    print("SECTION 1: BCH ERROR CORRECTION UNIT TEST")
    print("=" * 75)
    
    pipeline = BiometricPipeline(use_mock_blockchain=True)
    input_size = pipeline.input_size
    
    # BCH parameters
    bch_t = pipeline.fuzzy_extractor.t
    print(f"\nBCH Parameters: n={pipeline.fuzzy_extractor.n}, k={pipeline.fuzzy_extractor.k}, t={bch_t}")
    print(f"Max correctable errors: {bch_t} bits ({bch_t/pipeline.fuzzy_extractor.n*100:.1f}%)")
    
    # Enroll test users for BCH test
    print("\nEnrolling test users...")
    num_test_users = 20
    test_users = []
    for i in range(num_test_users):
        torch.manual_seed(i * 1000)
        image = torch.randn(1, 3, input_size[0], input_size[1])
        embedding = pipeline.extract_features(image)
        binary_code = pipeline.biohasher(embedding)
        binary_bytes = pipeline.biohasher.to_bytes(binary_code[0])
        secret_key, helper_data = pipeline.fuzzy_extractor.gen(binary_bytes)
        test_users.append({
            'binary_bytes': binary_bytes,
            'helper_data': helper_data,
            'secret_key': secret_key
        })
    
    # Test bit flip tolerance
    print("\nTesting BCH error correction with direct bit flips:")
    print("-" * 60)
    
    bit_flip_levels = [0, 5, 10, 15, 20, 25, 29, 30, 35, 40]
    bch_results = {}
    
    for num_flips in bit_flip_levels:
        successes = 0
        total_errors = 0
        num_attempts = 50
        
        for _ in range(num_attempts):
            user = test_users[np.random.randint(num_test_users)]
            
            # Flip bits directly
            noisy_binary = flip_bits(user['binary_bytes'], num_flips)
            
            # Try to recover
            recovered_key, errors = pipeline.fuzzy_extractor.rep(noisy_binary, user['helper_data'])
            
            if recovered_key is not None and recovered_key == user['secret_key']:
                successes += 1
                total_errors += errors
        
        frr = (num_attempts - successes) / num_attempts * 100
        avg_errors = total_errors / max(successes, 1)
        
        status = "✓" if num_flips <= bch_t else "✗"
        print(f"  {status} Bit flips {num_flips:2d}: FRR = {frr:5.1f}%, Avg errors corrected = {avg_errors:.1f}")
        
        bch_results[num_flips] = {'frr': frr, 'avg_errors': avg_errors}
    
    print("-" * 60)
    print(f"Expected: FRR = 0% for flips ≤ {bch_t}, FRR > 0% for flips > {bch_t}")
    
    # =========================================================================
    # SECTION 2: STANDARD BIOHASH EVALUATION
    # =========================================================================
    print("\n" + "=" * 75)
    print("SECTION 2: STANDARD BIOHASH EVALUATION (Embedding Noise)")
    print("=" * 75)
    
    num_users = 50
    num_auth_attempts = 100
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    print(f"\nParameters:")
    print(f"  Users: {num_users}")
    print(f"  Auth attempts per noise level: {num_auth_attempts}")
    
    # Enroll users
    print("\nEnrolling users...")
    users = []
    for i in range(num_users):
        torch.manual_seed(i)
        image = torch.randn(1, 3, input_size[0], input_size[1])
        secret_key, helper_data, keypair = pipeline.enroll(image)
        embedding = pipeline.extract_features(image)
        
        users.append({
            'image': image,
            'embedding': embedding,
            'helper_data': helper_data,
            'secret_key': secret_key
        })
    
    print(f"  Enrolled {num_users} users")
    
    # Test with embedding noise
    print("\nGenuine authentication with embedding noise:")
    print("-" * 60)
    
    standard_results = {}
    for noise_level in noise_levels:
        successes = 0
        total_errors = 0
        hamming_distances = []
        
        for _ in range(num_auth_attempts):
            user = users[np.random.randint(num_users)]
            
            # Add noise to embedding
            noisy_embedding = simulate_biometric_noise(user['embedding'], noise_level)
            
            # Get original binary for Hamming distance
            orig_binary = pipeline.biohasher(user['embedding'])
            noisy_binary = pipeline.biohasher(noisy_embedding)
            
            # Compute Hamming distance
            hamming = int(torch.sum(orig_binary != noisy_binary).item())
            hamming_distances.append(hamming)
            
            # Try to recover
            binary_bytes = pipeline.biohasher.to_bytes(noisy_binary[0])
            recovered_key, errors = pipeline.fuzzy_extractor.rep(binary_bytes, user['helper_data'])
            
            if recovered_key is not None:
                successes += 1
                total_errors += errors
        
        frr = (num_auth_attempts - successes) / num_auth_attempts * 100
        avg_errors = total_errors / max(successes, 1)
        avg_hamming = np.mean(hamming_distances)
        
        print(f"  Noise {noise_level*100:4.0f}%: FRR = {frr:5.1f}%, "
              f"Avg Hamming = {avg_hamming:.1f} bits, Avg errors corrected = {avg_errors:.1f}")
        
        standard_results[noise_level] = {
            'frr': frr, 
            'avg_hamming': avg_hamming,
            'avg_errors': avg_errors
        }
    
    # Impostor test
    print("\nImpostor detection:")
    print("-" * 60)
    
    false_accepts = 0
    impostor_hamming = []
    
    for _ in range(num_auth_attempts):
        victim = users[np.random.randint(num_users)]
        
        # Generate impostor
        impostor_embedding = torch.randn(1, 512)
        impostor_embedding = torch.nn.functional.normalize(impostor_embedding, p=2, dim=1)
        
        # Hamming distance
        orig_binary = pipeline.biohasher(victim['embedding'])
        imp_binary = pipeline.biohasher(impostor_embedding)
        hamming = int(torch.sum(orig_binary != imp_binary).item())
        impostor_hamming.append(hamming)
        
        # Try to authenticate
        binary_bytes = pipeline.biohasher.to_bytes(imp_binary[0])
        recovered_key, _ = pipeline.fuzzy_extractor.rep(binary_bytes, victim['helper_data'])
        
        if recovered_key is not None:
            false_accepts += 1
    
    far = false_accepts / num_auth_attempts * 100
    print(f"  FAR: {far:.3f}%")
    print(f"  Avg impostor Hamming: {np.mean(impostor_hamming):.1f} bits")
    
    # =========================================================================
    # SECTION 3: IMPROVED BIOHASH (Reliable Bit Selection)
    # =========================================================================
    print("\n" + "=" * 75)
    print("SECTION 3: IMPROVED BIOHASH (Reliable Bit Selection)")
    print("=" * 75)
    
    try:
        from biohashing_improved import ImprovedBioHasher, AdaptiveFuzzyExtractor
        
        # Initialize improved components
        improved_biohasher = ImprovedBioHasher(
            reliability_threshold=0.05,
            min_reliable_bits=200,
            use_median_threshold=True
        )
        improved_fe = AdaptiveFuzzyExtractor(bch_m=9, bch_t=29)
        
        print(f"\nConfiguration:")
        print(f"  Reliability threshold: 0.05")
        print(f"  Min reliable bits: 200")
        print(f"  BCH t: 29")
        
        # Enroll with improved system
        print("\nEnrolling users with improved BioHash...")
        improved_users = []
        reliable_bits_counts = []
        
        for i in range(num_users):
            torch.manual_seed(i)
            image = torch.randn(1, 3, input_size[0], input_size[1])
            embedding = pipeline.extract_features(image)
            
            # Use improved BioHasher
            binary_code, reliable_info = improved_biohasher(embedding)
            reliable_bits_counts.append(reliable_info.num_reliable)
            
            # Generate key
            key, helper_data = improved_fe.gen(binary_code)
            
            improved_users.append({
                'embedding': embedding,
                'binary_code': binary_code,
                'reliable_info': reliable_info,
                'helper_data': helper_data,
                'key': key
            })
        
        avg_reliable = np.mean(reliable_bits_counts)
        print(f"  Enrolled {num_users} users")
        print(f"  Avg reliable bits: {avg_reliable:.0f} / 511")
        
        # Test improved system
        print("\nGenuine authentication (same embedding, different noise):")
        print("-" * 60)
        
        improved_successes = 0
        improved_errors = []
        improved_hamming = []
        
        # Simulate intra-class variation with small noise
        for _ in range(num_auth_attempts):
            user = improved_users[np.random.randint(num_users)]
            
            # Small noise to simulate same person, different capture
            noisy_embedding = user['embedding'] + torch.randn_like(user['embedding']) * 0.02
            noisy_embedding = torch.nn.functional.normalize(noisy_embedding, p=2, dim=1)
            
            # Get binary with same reliable bits
            noisy_binary, _ = improved_biohasher(noisy_embedding, user['reliable_info'])
            
            # Hamming distance
            hamming = int(np.sum(user['binary_code'] != noisy_binary))
            improved_hamming.append(hamming)
            
            # Try to recover
            recovered_key, errors = improved_fe.rep(noisy_binary, user['helper_data'])
            
            if recovered_key is not None:
                improved_successes += 1
                improved_errors.append(errors)
        
        improved_frr = (num_auth_attempts - improved_successes) / num_auth_attempts * 100
        avg_improved_hamming = np.mean(improved_hamming)
        avg_improved_errors = np.mean(improved_errors) if improved_errors else 0
        
        print(f"  FRR: {improved_frr:.2f}%")
        print(f"  Avg Hamming distance: {avg_improved_hamming:.1f} bits ({avg_improved_hamming/avg_reliable*100:.1f}% of {avg_reliable:.0f})")
        print(f"  Avg errors corrected: {avg_improved_errors:.1f}")
        
        # Impostor test for improved system
        print("\nImpostor detection (improved):")
        print("-" * 60)
        
        improved_false_accepts = 0
        improved_impostor_hamming = []
        
        for _ in range(num_auth_attempts):
            victim = improved_users[np.random.randint(num_users)]
            
            # Generate impostor
            impostor_embedding = torch.randn(1, 512)
            impostor_embedding = torch.nn.functional.normalize(impostor_embedding, p=2, dim=1)
            
            # Use victim's reliable bits
            imp_binary, _ = improved_biohasher(impostor_embedding, victim['reliable_info'])
            
            hamming = int(np.sum(victim['binary_code'] != imp_binary))
            improved_impostor_hamming.append(hamming)
            
            recovered_key, _ = improved_fe.rep(imp_binary, victim['helper_data'])
            
            if recovered_key is not None:
                improved_false_accepts += 1
        
        improved_far = improved_false_accepts / num_auth_attempts * 100
        print(f"  FAR: {improved_far:.3f}%")
        print(f"  Avg impostor Hamming: {np.mean(improved_impostor_hamming):.1f} bits")
        
        has_improved = True
        
    except ImportError:
        print("\n  [SKIPPED] biohashing_improved.py not found")
        print("  Run: python evaluate_improved.py for improved evaluation")
        has_improved = False
        improved_frr = None
        avg_improved_hamming = None
    
    # =========================================================================
    # SECTION 4: PERFORMANCE BENCHMARK
    # =========================================================================
    print("\n" + "=" * 75)
    print("SECTION 4: PERFORMANCE BENCHMARK")
    print("=" * 75)
    
    # Enrollment time
    enroll_times = []
    for _ in range(50):
        image = torch.randn(1, 3, input_size[0], input_size[1])
        start = time.time()
        pipeline.enroll(image)
        enroll_times.append(time.time() - start)
    
    # Authentication time
    auth_times = []
    user = users[0]
    for _ in range(50):
        start = time.time()
        pipeline.authenticate(user['image'], user['helper_data'])
        auth_times.append(time.time() - start)
    
    print(f"\n  Enrollment:     {np.mean(enroll_times)*1000:.2f} ± {np.std(enroll_times)*1000:.2f} ms")
    print(f"  Authentication: {np.mean(auth_times)*1000:.2f} ± {np.std(auth_times)*1000:.2f} ms")
    
    # =========================================================================
    # SECTION 5: BLOCKCHAIN GAS COST EVALUATION
    # =========================================================================
    print("\n" + "=" * 75)
    print("SECTION 5: BLOCKCHAIN TRANSACTION FEE EVALUATION")
    print("=" * 75)
    
    # Gas estimates (based on contract analysis)
    gas_estimates = {
        'register': 194389,
        'requestChallenge': 67599,
        'authenticate': 76950,
        'getHelperData': 0,  # View function - FREE
    }
    
    gas_price_gwei = 20.0
    eth_price_usd = 3500.0
    
    print(f"\nParameters: Gas Price = {gas_price_gwei} Gwei, ETH = ${eth_price_usd:,.0f}")
    
    print("\nGas costs by operation:")
    print("-" * 70)
    print(f"{'Operation':<25} {'Gas':>12} {'ETH':>15} {'USD':>12}")
    print("-" * 70)
    
    for op, gas in gas_estimates.items():
        if gas > 0:
            eth = gas * gas_price_gwei * 1e-9
            usd = eth * eth_price_usd
            print(f"{op:<25} {gas:>12,} {eth:>15.8f} ${usd:>10.4f}")
        else:
            print(f"{op:<25} {'FREE':>12} {'FREE':>15} {'FREE':>12}")
    
    print("-" * 70)
    
    # Complete flow costs
    enroll_gas = gas_estimates['register']
    auth_gas = gas_estimates['requestChallenge'] + gas_estimates['authenticate']
    
    enroll_eth = enroll_gas * gas_price_gwei * 1e-9
    enroll_usd = enroll_eth * eth_price_usd
    auth_eth = auth_gas * gas_price_gwei * 1e-9
    auth_usd = auth_eth * eth_price_usd
    
    print(f"\nComplete flow costs:")
    print(f"  ENROLLMENT (one-time):  {enroll_gas:>10,} gas = {enroll_eth:.8f} ETH = ${enroll_usd:.4f}")
    print(f"  AUTH FLOW (per-login):  {auth_gas:>10,} gas = {auth_eth:.8f} ETH = ${auth_usd:.4f}")
    
    # Network comparison
    print("\nCost comparison across networks:")
    print("-" * 70)
    print(f"{'Network':<20} {'Gas Price':>12} {'Register':>15} {'Auth':>15}")
    print("-" * 70)
    
    networks = [
        ("Ethereum Mainnet", 20, 3500),
        ("Polygon", 50, 0.80),
        ("Arbitrum", 0.1, 3500),
        ("Optimism", 0.01, 3500),
        ("Local/Testnet", 0, 0),
    ]
    
    for network, gp, ep in networks:
        if ep == 0:
            print(f"{network:<20} {gp:>10} Gwei {'FREE':>15} {'FREE':>15}")
        else:
            reg_cost = enroll_gas * gp * 1e-9 * ep
            auth_cost = auth_gas * gp * 1e-9 * ep
            print(f"{network:<20} {gp:>10} Gwei ${reg_cost:>13.4f} ${auth_cost:>13.4f}")
    
    print("-" * 70)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 75)
    print("BENCHMARK SUMMARY")
    print("=" * 75)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│                         BIOMETRIC BENCHMARK RESULTS                      │
├─────────────────────────────────────────────────────────────────────────┤
│ [1] BCH ERROR CORRECTION                                                 │
│     • BCH({pipeline.fuzzy_extractor.n}, {pipeline.fuzzy_extractor.k}, {bch_t}) - corrects up to {bch_t} bit errors              │
│     • FRR @ {bch_t} flips: {bch_results.get(bch_t, {}).get('frr', 'N/A'):.1f}%                                              │
│     • FRR @ {bch_t+1} flips: {bch_results.get(bch_t+1, {}).get('frr', 'N/A'):.1f}%                                             │
├─────────────────────────────────────────────────────────────────────────┤
│ [2] STANDARD BIOHASH (511 bits)                                          │
│     • Genuine Hamming @ 0% noise: {standard_results[0.0]['avg_hamming']:.1f} bits                              │
│     • Genuine Hamming @ 5% noise: {standard_results[0.05]['avg_hamming']:.1f} bits                            │
│     • FRR @ 0% noise: {standard_results[0.0]['frr']:.1f}%                                              │
│     • FAR: {far:.3f}%                                                          │
├─────────────────────────────────────────────────────────────────────────┤""")
    
    if has_improved:
        print(f"""│ [3] IMPROVED BIOHASH ({avg_reliable:.0f} reliable bits)                                   │
│     • Genuine Hamming: {avg_improved_hamming:.1f} bits ({avg_improved_hamming/avg_reliable*100:.1f}%)                                   │
│     • FRR: {improved_frr:.2f}%                                                        │
│     • FAR: {improved_far:.3f}%                                                         │
├─────────────────────────────────────────────────────────────────────────┤""")
    
    print(f"""│ [4] PERFORMANCE                                                          │
│     • Enrollment: {np.mean(enroll_times)*1000:.2f} ms                                              │
│     • Authentication: {np.mean(auth_times)*1000:.2f} ms                                           │
├─────────────────────────────────────────────────────────────────────────┤
│ [5] BLOCKCHAIN COSTS (Ethereum Mainnet @ 20 Gwei)                        │
│     • Registration: ${enroll_usd:.2f}                                              │
│     • Authentication: ${auth_usd:.2f}                                             │
│     • Layer 2 (Polygon): ~$0.01                                          │
└─────────────────────────────────────────────────────────────────────────┘
    """)
    
    # Comparison table
    if has_improved:
        print("\n" + "-" * 75)
        print("STANDARD vs IMPROVED BIOHASH COMPARISON")
        print("-" * 75)
        print(f"{'Metric':<30} {'Standard':<20} {'Improved':<20}")
        print("-" * 75)
        
        std_hamming = standard_results[0.05]['avg_hamming']
        std_frr = standard_results[0.05]['frr']
        
        print(f"{'Bits used':<30} {'511':<20} {avg_reliable:<20.0f}")
        print(f"{'Genuine Hamming (bits)':<30} {'~' + str(int(std_hamming)):<20} {avg_improved_hamming:<20.1f}")
        print(f"{'Genuine Hamming (%)':<30} {'~' + str(int(std_hamming/511*100)) + '%':<20} {str(round(avg_improved_hamming/avg_reliable*100, 1)) + '%':<20}")
        print(f"{'FRR':<30} {str(int(std_frr)) + '%':<20} {str(round(improved_frr, 2)) + '%':<20}")
        print(f"{'FAR':<30} {str(round(far, 1)) + '%':<20} {str(round(improved_far, 1)) + '%':<20}")
        print("-" * 75)
    
    print("\n" + "=" * 75)
    print("Benchmark completed!")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(
        description="CNN + Fuzzy Extractor + Blockchain Biometric Authentication"
    )
    parser.add_argument(
        '--mode',
        choices=['demo', 'benchmark', 'train'],
        default='demo',
        help="Execution mode"
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help="Path to custom config file"
    )
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        run_demo()
    elif args.mode == 'benchmark':
        run_benchmark()
    elif args.mode == 'train':
        print("Training mode requires a dataset.")
        print("Please see the training documentation in README.md")
        sys.exit(1)


if __name__ == "__main__":
    main()