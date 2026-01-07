#!/usr/bin/env python3
"""
Diagnostic Script: Why is FRR = 100%?
=====================================

This script tests each component to find where the bug is.
"""

import torch
import numpy as np
import secrets

# Local imports
from model import BiometricEncoder
from biohashing import BioHasher
from fuzzy_extractor import FuzzyExtractor, BCHCode
from config import DEFAULT_CONFIG


def test_bch_directly():
    """Direct test of BCH encode/decode."""
    print("\n[0] DIRECT BCH ENCODE/DECODE TEST")
    print("-" * 50)
    
    bch = BCHCode(m=9, t=29)
    print(f"  BCH: n={bch.n}, k={bch.k}, t={bch.t}, ecc_bits={bch.ecc_bits}")
    
    # Check if bchlib has ecc_bytes attribute
    if bch.bch is not None:
        print(f"  bchlib.n = {bch.bch.n}")
        print(f"  bchlib.ecc_bits = {bch.bch.ecc_bits}")
        if hasattr(bch.bch, 'ecc_bytes'):
            print(f"  bchlib.ecc_bytes = {bch.bch.ecc_bytes}")
        else:
            print(f"  bchlib.ecc_bytes = NOT AVAILABLE (using ceil calculation)")
    
    # Generate random message
    msg_bytes = bch.k // 8
    original_msg = secrets.token_bytes(msg_bytes)
    print(f"  Original message: {msg_bytes} bytes")
    
    # Encode
    codeword = bch.encode(original_msg, debug=True)
    print(f"  Codeword length: {len(codeword)} bytes")
    
    # Decode immediately (no errors)
    recovered, nerrors = bch.decode(codeword, debug=True)
    
    if recovered is not None:
        match = recovered[:msg_bytes] == original_msg[:msg_bytes]
        print(f"  Recovery: {'✓ SUCCESS' if match else '✗ MISMATCH'}")
        print(f"  Errors corrected: {nerrors}")
        if not match:
            print(f"  Original: {original_msg[:16].hex()}...")
            print(f"  Recovered: {recovered[:16].hex()}...")
        return match
    else:
        print("  Recovery: ✗ FAILED (decode returned None)")
        return False


def test_encoder_determinism():
    """Test if encoder produces same output for same input."""
    print("\n[1] ENCODER DETERMINISM TEST")
    print("-" * 50)
    
    encoder = BiometricEncoder()
    input_size = encoder.input_size
    
    # Same input
    torch.manual_seed(42)
    image = torch.randn(1, 3, input_size[0], input_size[1])
    
    # Extract twice
    emb1 = encoder(image)
    emb2 = encoder(image)
    
    diff = (emb1 - emb2).abs().max().item()
    print(f"  Same image, embedding diff: {diff:.10f}")
    print(f"  Deterministic: {'✓ YES' if diff < 1e-6 else '✗ NO'}")
    
    return emb1, diff < 1e-6


def test_biohasher_determinism(embedding):
    """Test if biohasher produces same output for same input."""
    print("\n[2] BIOHASHER DETERMINISM TEST")
    print("-" * 50)
    
    biohasher = BioHasher(DEFAULT_CONFIG.biohash)
    
    # Binarize twice
    binary1 = biohasher(embedding)
    binary2 = biohasher(embedding)
    
    # Convert to bytes
    bytes1 = biohasher.to_bytes(binary1[0])
    bytes2 = biohasher.to_bytes(binary2[0])
    
    print(f"  Binary code length: {len(binary1[0])} bits")
    print(f"  Bytes representation: {len(bytes1)} bytes")
    print(f"  Same binary code: {'✓ YES' if bytes1 == bytes2 else '✗ NO'}")
    
    # Check bit distribution
    bits = binary1[0].numpy() if hasattr(binary1[0], 'numpy') else binary1[0]
    ones = np.sum(bits)
    print(f"  Bit distribution: {ones} ones, {len(bits) - ones} zeros ({ones/len(bits)*100:.1f}% ones)")
    
    return bytes1, bytes1 == bytes2


def test_fuzzy_extractor_roundtrip(binary_code):
    """Test if fuzzy extractor can recover key from same input."""
    print("\n[3] FUZZY EXTRACTOR ROUNDTRIP TEST")
    print("-" * 50)
    
    fe = FuzzyExtractor(DEFAULT_CONFIG.fuzzy_extractor)
    
    print(f"  Input biometric: {len(binary_code)} bytes")
    
    # Generate key and helper data
    key, helper_data = fe.gen(binary_code)
    print(f"  Generated key: {key.hex()[:32]}...")
    print(f"  Helper data size: {len(helper_data.to_bytes())} bytes")
    print(f"  Sketch size: {len(helper_data.sketch)} bytes (should match codeword)")
    
    # Recover with same input (should always work)
    recovered_key, errors = fe.rep(binary_code, helper_data)
    
    if recovered_key is not None:
        match = recovered_key == key
        print(f"  Recovered key: {recovered_key.hex()[:32]}...")
        print(f"  Keys match: {'✓ YES' if match else '✗ NO'}")
        print(f"  Errors corrected: {errors}")
        return match, helper_data
    else:
        print(f"  ✗ FAILED to recover key!")
        print(f"  This should NEVER happen with same input!")
        return False, helper_data


def test_full_pipeline():
    """Test the full enrollment -> authentication flow."""
    print("\n[4] FULL PIPELINE TEST")
    print("-" * 50)
    
    # Initialize components
    encoder = BiometricEncoder()
    biohasher = BioHasher(DEFAULT_CONFIG.biohash)
    fe = FuzzyExtractor(DEFAULT_CONFIG.fuzzy_extractor)
    
    input_size = encoder.input_size
    
    # ENROLLMENT
    torch.manual_seed(42)
    image = torch.randn(1, 3, input_size[0], input_size[1])
    
    print("  ENROLLMENT:")
    emb_enroll = encoder(image)
    print(f"    Embedding shape: {emb_enroll.shape}")
    print(f"    Embedding norm: {emb_enroll.norm().item():.6f}")
    
    binary_enroll = biohasher(emb_enroll)
    bytes_enroll = biohasher.to_bytes(binary_enroll[0])
    print(f"    Binary code: {len(bytes_enroll)} bytes")
    
    key, helper_data = fe.gen(bytes_enroll)
    print(f"    Key generated: {key.hex()[:16]}...")
    
    # AUTHENTICATION (same image)
    print("\n  AUTHENTICATION (same image):")
    emb_auth = encoder(image)  # Same image
    
    emb_diff = (emb_enroll - emb_auth).abs().max().item()
    print(f"    Embedding diff from enrollment: {emb_diff:.10f}")
    
    binary_auth = biohasher(emb_auth)
    bytes_auth = biohasher.to_bytes(binary_auth[0])
    
    # Compare binary codes
    bits_enroll = np.frombuffer(bytes_enroll, dtype=np.uint8)
    bits_auth = np.frombuffer(bytes_auth, dtype=np.uint8)
    
    # Count bit differences
    xor = np.bitwise_xor(bits_enroll, bits_auth)
    bit_diffs = sum(bin(b).count('1') for b in xor)
    print(f"    Binary code hamming distance: {bit_diffs} bits")
    
    # Try to recover
    recovered_key, errors = fe.rep(bytes_auth, helper_data)
    
    if recovered_key is not None:
        match = recovered_key == key
        print(f"    Key recovered: {'✓ YES' if match else '✗ NO'}")
        print(f"    Errors corrected: {errors}")
        return True
    else:
        print(f"    ✗ KEY RECOVERY FAILED!")
        print(f"    Hamming distance ({bit_diffs}) vs BCH correction capacity ({fe.t})")
        return False


def test_threshold_sensitivity(embedding):
    """Check how many values are near the binarization threshold (0)."""
    print("\n[5] THRESHOLD SENSITIVITY TEST")
    print("-" * 50)
    
    biohasher = BioHasher(DEFAULT_CONFIG.biohash)
    
    # Get the projection matrix and project
    projection = biohasher.projection_matrix
    projected = torch.matmul(embedding, projection.T)
    
    # Count values near threshold
    values = projected[0].numpy() if hasattr(projected[0], 'numpy') else projected[0]
    
    thresholds = [0.001, 0.01, 0.05, 0.1]
    print(f"  Projected values stats:")
    print(f"    Mean: {np.mean(values):.4f}")
    print(f"    Std:  {np.std(values):.4f}")
    print(f"    Min:  {np.min(values):.4f}")
    print(f"    Max:  {np.max(values):.4f}")
    
    print(f"\n  Values near threshold (0):")
    for thresh in thresholds:
        count = np.sum(np.abs(values) < thresh)
        pct = count / len(values) * 100
        print(f"    |value| < {thresh}: {count} bits ({pct:.1f}%)")
    
    # These near-zero values can flip with tiny embedding changes
    near_zero = np.sum(np.abs(values) < 0.01)
    print(f"\n  ⚠️  {near_zero} bits are unstable (|value| < 0.01)")
    print(f"  These bits may flip with tiny floating-point differences!")


def main():
    print("=" * 60)
    print("DIAGNOSTIC: Finding the 100% FRR Bug")
    print("=" * 60)
    
    # Test 0: Direct BCH test (most fundamental)
    bch_ok = test_bch_directly()
    
    if not bch_ok:
        print("\n" + "=" * 60)
        print("⚠️  CRITICAL: Direct BCH encode/decode failed!")
        print("The bug is in the BCH code implementation.")
        print("=" * 60)
        return
    
    # Test 1: Encoder determinism
    embedding, encoder_ok = test_encoder_determinism()
    
    # Test 2: Biohasher determinism
    binary_code, biohasher_ok = test_biohasher_determinism(embedding)
    
    # Test 3: Fuzzy extractor roundtrip
    fe_ok, helper_data = test_fuzzy_extractor_roundtrip(binary_code)
    
    # Test 4: Full pipeline
    pipeline_ok = test_full_pipeline()
    
    # Test 5: Threshold sensitivity
    test_threshold_sensitivity(embedding)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  BCH encode/decode:         {'✓' if bch_ok else '✗'}")
    print(f"  Encoder deterministic:     {'✓' if encoder_ok else '✗'}")
    print(f"  BioHasher deterministic:   {'✓' if biohasher_ok else '✗'}")
    print(f"  FuzzyExtractor roundtrip:  {'✓' if fe_ok else '✗'}")
    print(f"  Full pipeline works:       {'✓' if pipeline_ok else '✗'}")
    
    if not pipeline_ok:
        print("\n  ROOT CAUSE ANALYSIS:")
        if not bch_ok:
            print("    → BCH encode/decode is broken!")
        elif not encoder_ok:
            print("    → Encoder is non-deterministic!")
        elif not biohasher_ok:
            print("    → BioHasher is non-deterministic!")
        elif not fe_ok:
            print("    → FuzzyExtractor fails on identical input!")
        else:
            print("    → Floating-point precision issue at binarization threshold")
            print("    → Small embedding differences cause bit flips > BCH capacity")


if __name__ == "__main__":
    main()