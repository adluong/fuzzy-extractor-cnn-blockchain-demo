#!/usr/bin/env python3
"""
Evaluate with Improved BioHasher (Reliable Bit Selection)
==========================================================

This script evaluates the biometric system using reliable bit selection,
which significantly reduces intra-class Hamming distance.

Usage:
    python evaluate_improved.py --mode lfw
    python evaluate_improved.py --mode lfw --reliability 0.03  # More bits, less reliable
    python evaluate_improved.py --mode lfw --reliability 0.10  # Fewer bits, more reliable
"""

import argparse
import numpy as np
import torch
from tqdm import tqdm
from dataclasses import dataclass
from typing import Tuple, Optional

# Local imports
from model import BiometricEncoder
from biohashing_improved import ImprovedBioHasher, AdaptiveFuzzyExtractor, ReliableBitsInfo
from config import DEFAULT_CONFIG


@dataclass
class EvalResult:
    frr: float
    far: float
    genuine_hamming: float
    impostor_hamming: float
    num_reliable_bits: float
    genuine_attempts: int
    impostor_attempts: int


class ImprovedEvaluator:
    """Evaluator using improved BioHasher with reliable bit selection."""
    
    def __init__(
        self, 
        reliability_threshold: float = 0.05,
        min_reliable_bits: int = 200,
        bch_t: int = 29,
        verbose: bool = True
    ):
        self.verbose = verbose
        
        if verbose:
            print("Initializing Improved Biometric System...")
        
        # Encoder
        self.encoder = BiometricEncoder()
        self.input_size = self.encoder.input_size
        
        # Improved BioHasher
        self.biohasher = ImprovedBioHasher(
            config=DEFAULT_CONFIG.biohash,
            reliability_threshold=reliability_threshold,
            min_reliable_bits=min_reliable_bits,
            use_median_threshold=True
        )
        
        # Fuzzy Extractor
        self.fuzzy_extractor = AdaptiveFuzzyExtractor(bch_m=9, bch_t=bch_t)
        
        if verbose:
            print(f"  Encoder: {self.encoder.backend}")
            print(f"  Reliability threshold: {reliability_threshold}")
            print(f"  Min reliable bits: {min_reliable_bits}")
            print(f"  BCH t: {bch_t}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for encoder."""
        from PIL import Image
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        elif image.max() > 1.0:
            image = image / 255.0
        
        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=-1)
        
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize((self.input_size[1], self.input_size[0]), Image.BILINEAR)
        image = np.array(pil_image).astype(np.float32) / 255.0
        
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor.float()
    
    def enroll(self, image: np.ndarray) -> Tuple[bytes, bytes, np.ndarray, ReliableBitsInfo]:
        """Enroll: returns (key, helper_data, binary_code, reliable_info)."""
        tensor = self.preprocess(image)
        embedding = self.encoder(tensor)
        
        # Get binary code with reliable bit selection
        binary_code, reliable_info = self.biohasher(embedding)
        
        # Generate key
        key, helper_data = self.fuzzy_extractor.gen(binary_code)
        
        return key, helper_data, binary_code, reliable_info
    
    def authenticate(
        self, 
        image: np.ndarray, 
        helper_data: bytes,
        reliable_info: ReliableBitsInfo
    ) -> Tuple[bool, int, np.ndarray]:
        """Authenticate: returns (success, errors, binary_code)."""
        tensor = self.preprocess(image)
        embedding = self.encoder(tensor)
        
        # Get binary code using SAME reliable bits as enrollment
        binary_code, _ = self.biohasher(embedding, reliable_info)
        
        # Reproduce key
        recovered_key, errors = self.fuzzy_extractor.rep(binary_code, helper_data)
        
        success = recovered_key is not None
        return success, errors if errors >= 0 else 0, binary_code
    
    @staticmethod
    def hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
        return int(np.sum(a != b))


def download_lfw():
    """Download LFW dataset."""
    print("\nDownloading LFW dataset...")
    try:
        from sklearn.datasets import fetch_lfw_pairs
        lfw = fetch_lfw_pairs(subset='test', color=True, resize=1.0, download_if_missing=True)
        print(f"  Downloaded {len(lfw.pairs)} pairs")
        return lfw
    except Exception as e:
        print(f"  Error: {e}")
        return None


def evaluate_lfw(evaluator: ImprovedEvaluator, max_pairs: int = 500) -> EvalResult:
    """Evaluate on LFW dataset."""
    print("\n" + "=" * 70)
    print("IMPROVED BIOHASH EVALUATION (LFW Dataset)")
    print("=" * 70)
    
    lfw = download_lfw()
    if lfw is None:
        return None
    
    pairs = lfw.pairs
    labels = lfw.target
    
    num_pairs = min(len(labels), max_pairs)
    indices = np.random.choice(len(labels), num_pairs, replace=False)
    
    print(f"\nEvaluating {num_pairs} pairs...")
    
    # Statistics
    genuine_successes = 0
    genuine_attempts = 0
    impostor_accepts = 0
    impostor_attempts = 0
    genuine_hamming = []
    impostor_hamming = []
    genuine_errors = []
    reliable_bits_counts = []
    
    for idx in tqdm(indices, desc="Processing"):
        img1 = pairs[idx, 0]
        img2 = pairs[idx, 1]
        label = labels[idx]
        
        try:
            # Enroll with image 1
            key, helper_data, binary1, reliable_info = evaluator.enroll(img1)
            reliable_bits_counts.append(reliable_info.num_reliable)
            
            # Authenticate with image 2
            success, errors, binary2 = evaluator.authenticate(img2, helper_data, reliable_info)
            
            # Hamming distance
            hamming = evaluator.hamming_distance(binary1, binary2)
            
            if label == 1:  # Genuine
                genuine_attempts += 1
                genuine_hamming.append(hamming)
                if success:
                    genuine_successes += 1
                    genuine_errors.append(errors)
            else:  # Impostor
                impostor_attempts += 1
                impostor_hamming.append(hamming)
                if success:
                    impostor_accepts += 1
                    
        except Exception as e:
            continue
    
    # Compute metrics
    frr = (genuine_attempts - genuine_successes) / max(genuine_attempts, 1) * 100
    far = impostor_accepts / max(impostor_attempts, 1) * 100
    avg_genuine = np.mean(genuine_hamming) if genuine_hamming else 0
    avg_impostor = np.mean(impostor_hamming) if impostor_hamming else 0
    avg_reliable = np.mean(reliable_bits_counts) if reliable_bits_counts else 0
    
    # Print results
    print("\n" + "-" * 50)
    print("RESULTS (with Reliable Bit Selection)")
    print("-" * 50)
    
    print(f"\n[CONFIGURATION]")
    print(f"  Avg reliable bits used: {avg_reliable:.0f} / 511")
    print(f"  BCH capacity: {evaluator.fuzzy_extractor.bch_t} bits")
    
    print(f"\n[GENUINE PAIRS]")
    print(f"  Attempts: {genuine_attempts}")
    print(f"  Successes: {genuine_successes}")
    print(f"  FRR: {frr:.2f}%")
    print(f"  Avg Hamming distance: {avg_genuine:.1f} bits ({avg_genuine/avg_reliable*100:.1f}% of {avg_reliable:.0f})")
    if genuine_hamming:
        p25, p50, p75 = np.percentile(genuine_hamming, [25, 50, 75])
        print(f"  Hamming percentiles (25/50/75): {p25:.0f} / {p50:.0f} / {p75:.0f}")
    
    print(f"\n[IMPOSTOR PAIRS]")
    print(f"  Attempts: {impostor_attempts}")
    print(f"  False accepts: {impostor_accepts}")
    print(f"  FAR: {far:.2f}%")
    print(f"  Avg Hamming distance: {avg_impostor:.1f} bits")
    
    # Analysis
    print(f"\n[ANALYSIS]")
    if avg_genuine <= evaluator.fuzzy_extractor.bch_t:
        print(f"  ✓ Genuine Hamming ({avg_genuine:.1f}) ≤ BCH t ({evaluator.fuzzy_extractor.bch_t})")
        print(f"  ✓ System should work with current configuration!")
    else:
        needed_t = int(np.ceil(avg_genuine * 1.2))
        print(f"  ⚠ Genuine Hamming ({avg_genuine:.1f}) > BCH t ({evaluator.fuzzy_extractor.bch_t})")
        print(f"  → Try: --reliability 0.10 (stricter) or --bch-t {needed_t}")
    
    return EvalResult(
        frr=frr,
        far=far,
        genuine_hamming=avg_genuine,
        impostor_hamming=avg_impostor,
        num_reliable_bits=avg_reliable,
        genuine_attempts=genuine_attempts,
        impostor_attempts=impostor_attempts
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate with Improved BioHasher')
    parser.add_argument('--mode', choices=['lfw'], default='lfw')
    parser.add_argument('--max-pairs', type=int, default=500)
    parser.add_argument('--reliability', type=float, default=0.05,
                        help='Reliability threshold (higher = fewer, more stable bits)')
    parser.add_argument('--min-bits', type=int, default=200,
                        help='Minimum reliable bits to select')
    parser.add_argument('--bch-t', type=int, default=29,
                        help='BCH error correction capacity')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("IMPROVED BIOMETRIC EVALUATION")
    print("(with Reliable Bit Selection)")
    print("=" * 70)
    
    evaluator = ImprovedEvaluator(
        reliability_threshold=args.reliability,
        min_reliable_bits=args.min_bits,
        bch_t=args.bch_t,
        verbose=True
    )
    
    result = evaluate_lfw(evaluator, max_pairs=args.max_pairs)
    
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Reliable bits: {result.num_reliable_bits:.0f}")
    print(f"  Genuine Hamming: {result.genuine_hamming:.1f} bits ({result.genuine_hamming/result.num_reliable_bits*100:.1f}%)")
    print(f"  FRR: {result.frr:.2f}%")
    print(f"  FAR: {result.far:.2f}%")
    
    # Print benchmark table
    print("\n" + "-" * 70)
    print("BENCHMARK COMPARISON")
    print("-" * 70)
    print(f"{'Metric':<30} {'Standard BioHash':<20} {'Improved (Reliable)':<20}")
    print("-" * 70)
    print(f"{'Bits used':<30} {'511':<20} {f'{result.num_reliable_bits:.0f}':<20}")
    print(f"{'Genuine Hamming (bits)':<30} {'~114':<20} {f'{result.genuine_hamming:.1f}':<20}")
    print(f"{'Genuine Hamming (%)':<30} {'~22%':<20} {f'{result.genuine_hamming/result.num_reliable_bits*100:.1f}%':<20}")
    print(f"{'FRR':<30} {'100%':<20} {f'{result.frr:.2f}%':<20}")
    print(f"{'FAR':<30} {'0%':<20} {f'{result.far:.2f}%':<20}")
    print("-" * 70)
    
    if result.frr > 50:
        print(f"\n[TRY THESE SETTINGS]")
        print(f"  python evaluate_improved.py --reliability 0.10 --min-bits 150")
        print(f"  python evaluate_improved.py --reliability 0.15 --min-bits 127 --bch-t 50")


if __name__ == "__main__":
    main()