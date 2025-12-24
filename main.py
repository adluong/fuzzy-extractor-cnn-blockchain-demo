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
        use_mock_blockchain: bool = True
    ):
        self.config = config or DEFAULT_CONFIG
        
        # Initialize components
        print("Initializing Biometric Pipeline...")
        
        # 1. CNN Encoder
        print("  [1/4] Loading CNN encoder...")
        self.encoder = BiometricEncoder(self.config.cnn)
        self.encoder.eval()
        
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
            embedding = self.encoder(image)
        
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
    torch.manual_seed(42)  # For reproducibility
    original_image = torch.randn(1, 3, 112, 112)
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
    impostor_image = torch.randn(1, 3, 112, 112)
    
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


def run_benchmark():
    """
    Run comprehensive security benchmarks.
    """
    print("=" * 70)
    print("SECURITY BENCHMARK SUITE")
    print("=" * 70)
    
    pipeline = BiometricPipeline(use_mock_blockchain=True)
    
    # Parameters
    num_users = 50
    num_auth_attempts = 100
    noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
    
    print(f"\nBenchmark Parameters:")
    print(f"  Number of simulated users: {num_users}")
    print(f"  Authentication attempts per noise level: {num_auth_attempts}")
    print(f"  Noise levels: {noise_levels}")
    
    # Enroll users
    print("\n[ENROLLMENT PHASE]")
    users = []
    for i in range(num_users):
        torch.manual_seed(i)
        image = torch.randn(1, 3, 112, 112)
        secret_key, helper_data, keypair = pipeline.enroll(image)
        
        # Store original embedding for noise simulation
        embedding = pipeline.extract_features(image)
        
        users.append({
            'id': i,
            'image': image,
            'embedding': embedding,
            'helper_data': helper_data,
            'keypair': keypair
        })
    
    print(f"  Enrolled {num_users} users")
    
    # Genuine attempt benchmarks
    print("\n[GENUINE AUTHENTICATION BENCHMARK]")
    print("-" * 50)
    
    for noise_level in noise_levels:
        successes = 0
        total_errors = 0
        
        for _ in range(num_auth_attempts):
            # Pick a random user
            user = users[np.random.randint(num_users)]
            
            # Add noise
            noisy_embedding = simulate_biometric_noise(user['embedding'], noise_level)
            
            # Binarize directly (bypass CNN for speed)
            binary_code = pipeline.biohasher(noisy_embedding)
            binary_bytes = pipeline.biohasher.to_bytes(binary_code[0])
            
            # Recover key
            recovered_key, errors = pipeline.fuzzy_extractor.rep(
                binary_bytes, 
                user['helper_data']
            )
            
            if recovered_key is not None:
                successes += 1
                total_errors += errors
        
        frr = (num_auth_attempts - successes) / num_auth_attempts * 100
        avg_errors = total_errors / max(successes, 1)
        
        print(f"  Noise {noise_level*100:4.0f}%: FRR = {frr:5.1f}%, "
              f"Avg errors corrected = {avg_errors:.1f}")
    
    # Impostor attempt benchmarks
    print("\n[IMPOSTOR DETECTION BENCHMARK]")
    print("-" * 50)
    
    false_accepts = 0
    for _ in range(num_auth_attempts):
        # Pick a victim
        victim = users[np.random.randint(num_users)]
        
        # Generate impostor biometric
        impostor_embedding = torch.randn(1, 512)
        impostor_embedding = torch.nn.functional.normalize(impostor_embedding, p=2, dim=1)
        
        # Attempt authentication
        binary_code = pipeline.biohasher(impostor_embedding)
        binary_bytes = pipeline.biohasher.to_bytes(binary_code[0])
        
        recovered_key, _ = pipeline.fuzzy_extractor.rep(
            binary_bytes,
            victim['helper_data']
        )
        
        if recovered_key is not None:
            false_accepts += 1
    
    far = false_accepts / num_auth_attempts * 100
    print(f"  False Acceptance Rate (FAR): {far:.3f}%")
    print(f"  False accepts: {false_accepts}/{num_auth_attempts}")
    
    # Performance benchmark
    print("\n[PERFORMANCE BENCHMARK]")
    print("-" * 50)
    
    # Enrollment time
    times = []
    for _ in range(100):
        image = torch.randn(1, 3, 112, 112)
        start = time.time()
        pipeline.enroll(image)
        times.append(time.time() - start)
    
    print(f"  Enrollment: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
    
    # Authentication time
    times = []
    user = users[0]
    for _ in range(100):
        start = time.time()
        pipeline.authenticate(user['image'], user['helper_data'])
        times.append(time.time() - start)
    
    print(f"  Authentication: {np.mean(times)*1000:.2f} ± {np.std(times)*1000:.2f} ms")
    
    print("\n" + "=" * 70)
    print("Benchmark completed!")
    print("=" * 70)


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