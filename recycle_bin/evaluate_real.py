#!/usr/bin/env python3
"""
Comprehensive Biometric Authentication Evaluation
==================================================

This script provides two evaluation modes:

1. BINARY NOISE TEST (Unit Test)
   - Directly flips bits in the binary code
   - Tests BCH error correction capacity
   - Fast, deterministic, no dataset required

2. REAL FACE EVALUATION (Publication-Ready)
   - Uses LFW (Labeled Faces in the Wild) dataset
   - Tests genuine pairs (same person, different images)
   - Tests impostor pairs (different people)
   - Computes FRR, FAR, EER metrics

Usage:
    python evaluate_real.py --mode binary    # Quick BCH unit test
    python evaluate_real.py --mode lfw       # Full LFW evaluation
    python evaluate_real.py --mode both      # Run both
"""

import os
import sys
import argparse
import numpy as np
import torch
from typing import Tuple, List, Optional
from dataclasses import dataclass
from tqdm import tqdm
import time

# Local imports
from model import BiometricEncoder
from biohashing import BioHasher
from fuzzy_extractor import FuzzyExtractor
from config import DEFAULT_CONFIG

# Try to import MTCNN for face alignment
try:
    from facenet_pytorch import MTCNN
    HAS_MTCNN = True
except ImportError:
    HAS_MTCNN = False
    print("Warning: MTCNN not available. Face alignment disabled.")


@dataclass
class EvaluationResult:
    """Stores evaluation metrics."""
    frr: float  # False Rejection Rate
    far: float  # False Acceptance Rate
    eer: float  # Equal Error Rate (optional)
    genuine_attempts: int
    genuine_successes: int
    impostor_attempts: int
    impostor_accepts: int
    avg_errors_corrected: float
    avg_hamming_distance_genuine: float
    avg_hamming_distance_impostor: float


class FaceAligner:
    """MTCNN-based face detection and alignment."""
    
    def __init__(self, image_size: int = 160, device: str = None):
        if not HAS_MTCNN:
            self.mtcnn = None
            return
            
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=20,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            post_process=True,  # Normalize to [-1, 1]
            device=device
        )
        self.image_size = image_size
    
    def align(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Detect and align face in image.
        
        Args:
            image: RGB image as numpy array (H, W, 3), values in [0, 255] or [0, 1]
            
        Returns:
            Aligned face tensor (1, 3, image_size, image_size) or None if no face detected
        """
        if self.mtcnn is None:
            return None
        
        # Ensure uint8 format for MTCNN
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # Detect and align
        try:
            from PIL import Image
            pil_image = Image.fromarray(image)
            aligned = self.mtcnn(pil_image)
            
            if aligned is not None:
                # MTCNN returns (3, H, W) tensor normalized to [-1, 1]
                # Convert to (1, 3, H, W) and to [0, 1] range for our pipeline
                aligned = (aligned + 1) / 2  # [-1, 1] -> [0, 1]
                aligned = aligned.unsqueeze(0)
                return aligned
        except Exception as e:
            pass
        
        return None


class BiometricEvaluator:
    """Evaluator for biometric authentication system."""
    
    def __init__(self, verbose: bool = True, use_alignment: bool = True):
        self.verbose = verbose
        self.use_alignment = use_alignment and HAS_MTCNN
        
        # Initialize components
        if verbose:
            print("Initializing components...")
        
        self.encoder = BiometricEncoder()
        self.biohasher = BioHasher(DEFAULT_CONFIG.biohash)
        self.fuzzy_extractor = FuzzyExtractor(DEFAULT_CONFIG.fuzzy_extractor)
        
        self.input_size = self.encoder.input_size
        
        # Initialize face aligner
        if self.use_alignment:
            self.aligner = FaceAligner(image_size=self.input_size[0])
            if verbose:
                print(f"  Face alignment: ENABLED (MTCNN)")
        else:
            self.aligner = None
            if verbose:
                print(f"  Face alignment: DISABLED")
        
        if verbose:
            print(f"  Encoder: {self.encoder.backend}")
            print(f"  Input size: {self.input_size}")
            print(f"  BCH: ({self.fuzzy_extractor.n}, {self.fuzzy_extractor.k}, {self.fuzzy_extractor.t})")
    
    def preprocess_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess image: detect face, align, and convert to tensor.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Preprocessed tensor or None if face detection failed
        """
        # Try MTCNN alignment first
        if self.aligner is not None:
            aligned = self.aligner.align(image)
            if aligned is not None:
                return aligned
        
        # Fallback: simple resize without alignment
        return self._simple_preprocess(image)
    
    def _simple_preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Simple preprocessing without face alignment."""
        from PIL import Image
        
        # Normalize to [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image / 255.0
        
        # Handle grayscale
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        
        # Resize
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
        image = np.array(pil_image).astype(np.float32) / 255.0
        
        # Convert to tensor (1, 3, H, W)
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.float()
    
    def extract_binary(self, image: torch.Tensor) -> bytes:
        """Extract binary code from image tensor."""
        embedding = self.encoder(image)
        binary = self.biohasher(embedding)
        return self.biohasher.to_bytes(binary[0])
    
    def extract_binary_from_numpy(self, image: np.ndarray) -> Optional[bytes]:
        """Extract binary code from numpy image with preprocessing."""
        tensor = self.preprocess_image(image)
        if tensor is None:
            return None
        return self.extract_binary(tensor)
    
    def enroll(self, image: torch.Tensor) -> Tuple[bytes, object, bytes]:
        """Enroll a biometric, return (key, helper_data, binary_code)."""
        binary_code = self.extract_binary(image)
        key, helper_data = self.fuzzy_extractor.gen(binary_code)
        return key, helper_data, binary_code
    
    def enroll_from_numpy(self, image: np.ndarray) -> Optional[Tuple[bytes, object, bytes]]:
        """Enroll from numpy image with preprocessing."""
        tensor = self.preprocess_image(image)
        if tensor is None:
            return None
        return self.enroll(tensor)
    
    def authenticate(self, image: torch.Tensor, helper_data: object) -> Tuple[bool, int, bytes]:
        """Authenticate, return (success, errors_corrected, binary_code)."""
        binary_code = self.extract_binary(image)
        recovered_key, errors = self.fuzzy_extractor.rep(binary_code, helper_data)
        success = recovered_key is not None
        return success, errors if errors >= 0 else 0, binary_code
    
    def authenticate_from_numpy(self, image: np.ndarray, helper_data: object) -> Optional[Tuple[bool, int, bytes]]:
        """Authenticate from numpy image with preprocessing."""
        tensor = self.preprocess_image(image)
        if tensor is None:
            return None
        return self.authenticate(tensor, helper_data)
    
    @staticmethod
    def hamming_distance(a: bytes, b: bytes) -> int:
        """Compute Hamming distance between two byte arrays."""
        a_arr = np.frombuffer(a, dtype=np.uint8)
        b_arr = np.frombuffer(b, dtype=np.uint8)
        xor = np.bitwise_xor(a_arr, b_arr)
        return sum(bin(byte).count('1') for byte in xor)
    
    @staticmethod
    def flip_bits(binary_code: bytes, num_flips: int) -> bytes:
        """Flip exactly num_flips random bits in binary code."""
        code = bytearray(binary_code)
        num_bits = len(code) * 8
        
        if num_flips > num_bits:
            num_flips = num_bits
        
        flip_positions = np.random.choice(num_bits, num_flips, replace=False)
        
        for pos in flip_positions:
            byte_idx = pos // 8
            bit_idx = pos % 8
            code[byte_idx] ^= (1 << bit_idx)
        
        return bytes(code)


# =============================================================================
# OPTION 1: BINARY-LEVEL NOISE TEST
# =============================================================================

def evaluate_binary_noise(evaluator: BiometricEvaluator, 
                          num_users: int = 20,
                          attempts_per_level: int = 50) -> dict:
    """
    Test BCH error correction with direct bit flips.
    
    This is a unit test for the fuzzy extractor, not a real-world evaluation.
    """
    print("\n" + "=" * 70)
    print("BINARY-LEVEL NOISE TEST (BCH Unit Test)")
    print("=" * 70)
    
    # Bit flip levels to test
    flip_levels = [0, 5, 10, 15, 20, 25, 29, 30, 35, 40]
    
    results = {}
    input_size = evaluator.input_size
    
    # Enroll users
    print(f"\nEnrolling {num_users} users...")
    users = []
    for i in range(num_users):
        torch.manual_seed(i * 1000)
        image = torch.randn(1, 3, input_size[0], input_size[1])
        key, helper_data, binary_code = evaluator.enroll(image)
        users.append({
            'key': key,
            'helper_data': helper_data,
            'binary_code': binary_code
        })
    
    print(f"\nTesting bit flip tolerance (BCH t={evaluator.fuzzy_extractor.t}):")
    print("-" * 50)
    
    for num_flips in flip_levels:
        successes = 0
        total_errors = 0
        
        for _ in range(attempts_per_level):
            user = users[np.random.randint(num_users)]
            
            # Flip bits directly in binary code
            noisy_binary = evaluator.flip_bits(user['binary_code'], num_flips)
            
            # Try to recover key
            recovered_key, errors = evaluator.fuzzy_extractor.rep(
                noisy_binary, 
                user['helper_data']
            )
            
            if recovered_key is not None and recovered_key == user['key']:
                successes += 1
                total_errors += errors
        
        frr = (attempts_per_level - successes) / attempts_per_level * 100
        avg_errors = total_errors / max(successes, 1)
        
        status = "✓" if num_flips <= evaluator.fuzzy_extractor.t else "✗"
        print(f"  {status} Bit flips {num_flips:2d}: FRR = {frr:5.1f}%, "
              f"Avg errors corrected = {avg_errors:.1f}")
        
        results[num_flips] = {
            'frr': frr,
            'successes': successes,
            'attempts': attempts_per_level,
            'avg_errors': avg_errors
        }
    
    print("-" * 50)
    print(f"BCH correction capacity: t = {evaluator.fuzzy_extractor.t} bits")
    print(f"Expected: FRR = 0% for flips ≤ {evaluator.fuzzy_extractor.t}, "
          f"FRR = 100% for flips > {evaluator.fuzzy_extractor.t}")
    
    return results


# =============================================================================
# OPTION 2: REAL FACE DATASET (LFW)
# =============================================================================

def download_lfw_dataset(data_dir: str = './data/lfw') -> str:
    """Download LFW dataset using sklearn."""
    print("\nDownloading LFW dataset (this may take a few minutes)...")
    
    try:
        from sklearn.datasets import fetch_lfw_pairs
        
        # Fetch LFW pairs (official evaluation pairs)
        lfw_pairs = fetch_lfw_pairs(
            data_home=data_dir,
            subset='test',  # Use test set for evaluation
            color=True,
            resize=1.0,
            download_if_missing=True
        )
        
        print(f"  Downloaded {len(lfw_pairs.pairs)} pairs")
        print(f"  Image shape: {lfw_pairs.pairs.shape[1:]}")
        print(f"  Labels: {np.bincount(lfw_pairs.target)}")
        
        return lfw_pairs
        
    except Exception as e:
        print(f"  Error downloading LFW: {e}")
        print("  Trying alternative method with torchvision...")
        
        try:
            from torchvision.datasets import LFWPairs
            
            dataset = LFWPairs(
                root=data_dir,
                split='test',
                download=True
            )
            
            return dataset
            
        except Exception as e2:
            print(f"  Alternative method also failed: {e2}")
            return None


def preprocess_lfw_image(image: np.ndarray, target_size: Tuple[int, int]) -> torch.Tensor:
    """Preprocess LFW image for FaceNet."""
    # LFW images are (H, W, C) in range [0, 255]
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0
    elif image.max() > 1.0:
        image = image / 255.0
    
    # Resize
    from PIL import Image
    if isinstance(image, np.ndarray):
        # Handle different shapes
        if len(image.shape) == 2:
            # Grayscale - convert to RGB
            image = np.stack([image, image, image], axis=-1)
        
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        image = np.array(pil_image).astype(np.float32) / 255.0
    
    # Convert to tensor (C, H, W)
    if len(image.shape) == 3:
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    else:
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
        tensor = tensor.repeat(1, 3, 1, 1)
    
    return tensor.float()


def evaluate_lfw(evaluator: BiometricEvaluator,
                 data_dir: str = './data/lfw',
                 max_pairs: int = 500) -> EvaluationResult:
    """
    Evaluate on LFW dataset.
    
    LFW provides official pairs:
    - Positive pairs: same person, different images
    - Negative pairs: different people
    """
    print("\n" + "=" * 70)
    print("REAL FACE EVALUATION (LFW Dataset)")
    print("=" * 70)
    
    # Download dataset
    lfw_data = download_lfw_dataset(data_dir)
    
    if lfw_data is None:
        print("\nFailed to download LFW. Using synthetic face pairs instead.")
        return evaluate_synthetic_faces(evaluator, max_pairs)
    
    # Check data format
    if hasattr(lfw_data, 'pairs'):
        # sklearn format
        pairs = lfw_data.pairs
        labels = lfw_data.target
    else:
        # torchvision format
        print("Processing torchvision LFW format...")
        pairs = []
        labels = []
        for i, (img1, img2, label) in enumerate(lfw_data):
            if i >= max_pairs:
                break
            pairs.append((np.array(img1), np.array(img2)))
            labels.append(label)
        pairs = np.array(pairs)
        labels = np.array(labels)
    
    # Limit number of pairs
    num_pairs = min(len(labels), max_pairs)
    indices = np.random.choice(len(labels), num_pairs, replace=False)
    
    alignment_status = "ENABLED (MTCNN)" if evaluator.use_alignment else "DISABLED"
    print(f"\nFace alignment: {alignment_status}")
    print(f"Evaluating {num_pairs} pairs...")
    
    # Statistics
    genuine_successes = 0
    genuine_attempts = 0
    impostor_accepts = 0
    impostor_attempts = 0
    genuine_errors = []
    genuine_hamming = []
    impostor_hamming = []
    
    # Track alignment failures
    alignment_failures = 0
    
    for idx in tqdm(indices, desc="Processing pairs"):
        if hasattr(lfw_data, 'pairs'):
            img1 = pairs[idx, 0]
            img2 = pairs[idx, 1]
        else:
            img1, img2 = pairs[idx]
        
        label = labels[idx]
        
        # Enroll with image 1 (using preprocessing with optional alignment)
        try:
            result1 = evaluator.enroll_from_numpy(img1)
            if result1 is None:
                alignment_failures += 1
                continue
            key, helper_data, binary1 = result1
        except Exception as e:
            continue
        
        # Authenticate with image 2
        try:
            result2 = evaluator.authenticate_from_numpy(img2, helper_data)
            if result2 is None:
                alignment_failures += 1
                continue
            success, errors, binary2 = result2
        except Exception as e:
            continue
        
        # Compute Hamming distance
        hamming = evaluator.hamming_distance(binary1, binary2)
        
        if label == 1:  # Same person (genuine pair)
            genuine_attempts += 1
            genuine_hamming.append(hamming)
            if success:
                genuine_successes += 1
                genuine_errors.append(errors)
        else:  # Different people (impostor pair)
            impostor_attempts += 1
            impostor_hamming.append(hamming)
            if success:
                impostor_accepts += 1
    
    # Compute metrics
    frr = (genuine_attempts - genuine_successes) / max(genuine_attempts, 1) * 100
    far = impostor_accepts / max(impostor_attempts, 1) * 100
    avg_errors = np.mean(genuine_errors) if genuine_errors else 0
    avg_genuine_hamming = np.mean(genuine_hamming) if genuine_hamming else 0
    avg_impostor_hamming = np.mean(impostor_hamming) if impostor_hamming else 0
    
    # Print results
    print("\n" + "-" * 50)
    print("RESULTS")
    print("-" * 50)
    
    if alignment_failures > 0:
        print(f"\n[PREPROCESSING]")
        print(f"  Face alignment failures: {alignment_failures}")
    
    print(f"\n[GENUINE PAIRS] (Same Person, Different Images)")
    print(f"  Attempts: {genuine_attempts}")
    print(f"  Successes: {genuine_successes}")
    print(f"  FRR (False Rejection Rate): {frr:.2f}%")
    print(f"  Avg Hamming distance: {avg_genuine_hamming:.1f} bits")
    print(f"  Avg errors corrected: {avg_errors:.1f}")
    
    # Show Hamming distance distribution
    if genuine_hamming:
        print(f"  Hamming distance range: [{min(genuine_hamming)}, {max(genuine_hamming)}]")
        percentiles = np.percentile(genuine_hamming, [25, 50, 75])
        print(f"  Hamming distance percentiles (25/50/75): {percentiles[0]:.0f} / {percentiles[1]:.0f} / {percentiles[2]:.0f}")
    
    print(f"\n[IMPOSTOR PAIRS] (Different People)")
    print(f"  Attempts: {impostor_attempts}")
    print(f"  False accepts: {impostor_accepts}")
    print(f"  FAR (False Acceptance Rate): {far:.2f}%")
    print(f"  Avg Hamming distance: {avg_impostor_hamming:.1f} bits")
    
    print(f"\n[ANALYSIS]")
    print(f"  BCH correction capacity: {evaluator.fuzzy_extractor.t} bits")
    print(f"  Genuine Hamming ({avg_genuine_hamming:.1f}) vs BCH t ({evaluator.fuzzy_extractor.t}):")
    if avg_genuine_hamming <= evaluator.fuzzy_extractor.t:
        print(f"    ✓ Within BCH capacity - good intra-class stability")
    elif avg_genuine_hamming <= evaluator.fuzzy_extractor.t * 2:
        print(f"    ⚠ Close to BCH capacity - consider increasing t or improving alignment")
    else:
        print(f"    ✗ Exceeds BCH capacity - need larger t or better encoder")
    
    print(f"  Impostor Hamming ({avg_impostor_hamming:.1f}) vs threshold:")
    if avg_impostor_hamming > evaluator.fuzzy_extractor.t * 2:
        print(f"    ✓ Well separated from genuine - good inter-class distance")
    else:
        print(f"    ⚠ Close to genuine range - risk of false accepts")
    
    result = EvaluationResult(
        frr=frr,
        far=far,
        eer=0.0,  # Would need threshold sweep to compute
        genuine_attempts=genuine_attempts,
        genuine_successes=genuine_successes,
        impostor_attempts=impostor_attempts,
        impostor_accepts=impostor_accepts,
        avg_errors_corrected=avg_errors,
        avg_hamming_distance_genuine=avg_genuine_hamming,
        avg_hamming_distance_impostor=avg_impostor_hamming
    )
    
    return result


def evaluate_synthetic_faces(evaluator: BiometricEvaluator, 
                             num_pairs: int = 500) -> EvaluationResult:
    """
    Fallback: Use synthetic variations to simulate face pairs.
    
    This simulates real face variation by:
    - Genuine: Same base image + small perturbation
    - Impostor: Different base images
    """
    print("\n" + "-" * 50)
    print("SYNTHETIC FACE EVALUATION (Fallback)")
    print("-" * 50)
    
    input_size = evaluator.input_size
    
    # Create "identities" (different random seeds)
    num_identities = 100
    
    genuine_successes = 0
    genuine_attempts = 0
    impostor_accepts = 0
    impostor_attempts = 0
    genuine_hamming = []
    impostor_hamming = []
    genuine_errors = []
    
    print(f"\nTesting {num_pairs} pairs...")
    
    for i in tqdm(range(num_pairs)):
        is_genuine = (i % 2 == 0)
        
        if is_genuine:
            # Same identity, different "capture"
            identity = np.random.randint(num_identities)
            
            # Image 1: base
            torch.manual_seed(identity * 1000)
            img1 = torch.randn(1, 3, input_size[0], input_size[1])
            
            # Image 2: same base + small variation
            torch.manual_seed(identity * 1000)
            img2_base = torch.randn(1, 3, input_size[0], input_size[1])
            # Add small noise to simulate capture variation
            variation = torch.randn_like(img2_base) * 0.01  # Very small!
            img2 = img2_base + variation
            img2 = torch.nn.functional.normalize(img2.view(1, -1), dim=1).view(img2.shape)
            
        else:
            # Different identities
            id1 = np.random.randint(num_identities)
            id2 = (id1 + np.random.randint(1, num_identities)) % num_identities
            
            torch.manual_seed(id1 * 1000)
            img1 = torch.randn(1, 3, input_size[0], input_size[1])
            
            torch.manual_seed(id2 * 1000)
            img2 = torch.randn(1, 3, input_size[0], input_size[1])
        
        # Enroll and authenticate
        try:
            key, helper_data, binary1 = evaluator.enroll(img1)
            success, errors, binary2 = evaluator.authenticate(img2, helper_data)
            hamming = evaluator.hamming_distance(binary1, binary2)
        except:
            continue
        
        if is_genuine:
            genuine_attempts += 1
            genuine_hamming.append(hamming)
            if success:
                genuine_successes += 1
                genuine_errors.append(errors)
        else:
            impostor_attempts += 1
            impostor_hamming.append(hamming)
            if success:
                impostor_accepts += 1
    
    # Compute metrics
    frr = (genuine_attempts - genuine_successes) / max(genuine_attempts, 1) * 100
    far = impostor_accepts / max(impostor_attempts, 1) * 100
    
    print(f"\n[SYNTHETIC RESULTS]")
    print(f"  Genuine pairs: FRR = {frr:.2f}%")
    print(f"  Impostor pairs: FAR = {far:.2f}%")
    print(f"  Avg genuine Hamming: {np.mean(genuine_hamming):.1f} bits")
    print(f"  Avg impostor Hamming: {np.mean(impostor_hamming):.1f} bits")
    
    return EvaluationResult(
        frr=frr,
        far=far,
        eer=0.0,
        genuine_attempts=genuine_attempts,
        genuine_successes=genuine_successes,
        impostor_attempts=impostor_attempts,
        impostor_accepts=impostor_accepts,
        avg_errors_corrected=np.mean(genuine_errors) if genuine_errors else 0,
        avg_hamming_distance_genuine=np.mean(genuine_hamming) if genuine_hamming else 0,
        avg_hamming_distance_impostor=np.mean(impostor_hamming) if impostor_hamming else 0
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate biometric authentication system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python evaluate_real.py --mode binary   # Quick BCH unit test
  python evaluate_real.py --mode lfw      # Full LFW evaluation  
  python evaluate_real.py --mode both     # Run both evaluations
  python evaluate_real.py --mode lfw --max-pairs 1000  # More pairs
  python evaluate_real.py --mode lfw --no-align  # Disable face alignment
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['binary', 'lfw', 'both'],
        default='both',
        help='Evaluation mode (default: both)'
    )
    
    parser.add_argument(
        '--max-pairs',
        type=int,
        default=500,
        help='Maximum number of LFW pairs to evaluate (default: 500)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='./data/lfw',
        help='Directory for LFW dataset (default: ./data/lfw)'
    )
    
    parser.add_argument(
        '--no-align',
        action='store_true',
        help='Disable MTCNN face alignment (default: enabled)'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("BIOMETRIC AUTHENTICATION EVALUATION")
    print("=" * 70)
    
    # Initialize evaluator
    use_alignment = not args.no_align
    evaluator = BiometricEvaluator(verbose=True, use_alignment=use_alignment)
    
    results = {}
    
    # Run evaluations
    if args.mode in ['binary', 'both']:
        results['binary'] = evaluate_binary_noise(evaluator)
    
    if args.mode in ['lfw', 'both']:
        results['lfw'] = evaluate_lfw(
            evaluator, 
            data_dir=args.data_dir,
            max_pairs=args.max_pairs
        )
    
    # Summary
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    if 'binary' in results:
        print("\n[Binary Noise Test]")
        print(f"  BCH capacity: t = {evaluator.fuzzy_extractor.t} bits")
        print(f"  FRR @ t bits: {results['binary'].get(evaluator.fuzzy_extractor.t, {}).get('frr', 'N/A')}%")
        print(f"  FRR @ t+1 bits: {results['binary'].get(evaluator.fuzzy_extractor.t + 1, {}).get('frr', 'N/A')}%")
    
    if 'lfw' in results:
        lfw = results['lfw']
        print("\n[LFW Real Face Test]")
        print(f"  Face alignment: {'ENABLED' if use_alignment else 'DISABLED'}")
        print(f"  False Rejection Rate (FRR): {lfw.frr:.2f}%")
        print(f"  False Acceptance Rate (FAR): {lfw.far:.2f}%")
        print(f"  Genuine Hamming distance: {lfw.avg_hamming_distance_genuine:.1f} bits")
        print(f"  Impostor Hamming distance: {lfw.avg_hamming_distance_impostor:.1f} bits")
        
        # Recommendation
        print("\n[RECOMMENDATION]")
        if lfw.avg_hamming_distance_genuine <= evaluator.fuzzy_extractor.t:
            print(f"  ✓ System is well-tuned. Genuine distance ({lfw.avg_hamming_distance_genuine:.1f}) ≤ BCH t ({evaluator.fuzzy_extractor.t})")
        else:
            needed_t = int(np.ceil(lfw.avg_hamming_distance_genuine * 1.2))  # 20% margin
            print(f"  ⚠ Genuine distance ({lfw.avg_hamming_distance_genuine:.1f}) > BCH t ({evaluator.fuzzy_extractor.t})")
            print(f"  → Consider using BCH with t ≥ {needed_t}")
            print(f"  → Or improve face alignment / encoder quality")
    
    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()