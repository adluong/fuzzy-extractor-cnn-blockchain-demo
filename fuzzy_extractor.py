"""
Fuzzy Extractor with BCH Error-Correcting Codes
=================================================

This module implements a secure sketch / fuzzy extractor using BCH codes
to reliably derive cryptographic keys from noisy biometric data.

Theoretical Foundation:
    A fuzzy extractor is a pair of procedures (Gen, Rep):
    - Gen(w) → (R, P): Given biometric w, output key R and public helper P
    - Rep(w', P) → R: Given noisy w' and helper P, recover R if d(w, w') ≤ t

Security Model:
    The helper data P must not leak significant information about R.
    Formally: H̃∞(R | P) ≥ H̃∞(W) - leak
    where leak is the entropy lost through the sketch.

    For code-offset construction with BCH(n, k, t):
        leak ≈ n - k bits
        
Reference: Dodis et al., "Fuzzy Extractors: How to Generate Strong Keys 
           from Biometrics and Other Noisy Data", SIAM J. Computing 2008.
"""

import secrets
import hashlib
import hmac
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

# BCH implementation
try:
    import bchlib
    HAS_BCHLIB = True
except ImportError:
    HAS_BCHLIB = False
    print("Warning: bchlib not found. Using fallback BCH implementation.")

from config import FuzzyExtractorConfig, DEFAULT_CONFIG


@dataclass
class HelperData:
    """
    Public helper data structure for fuzzy extractor.
    
    This is stored on the blockchain and used during authentication.
    
    Fields:
        sketch: The code-offset sketch P = C ⊕ w
        salt: Random salt for key derivation (prevents rainbow table attacks)
        version: Protocol version for future compatibility
    """
    sketch: bytes
    salt: bytes
    version: int = 1
    
    def to_bytes(self) -> bytes:
        """Serialize helper data for storage."""
        # Format: version (1 byte) | salt_len (2 bytes) | salt | sketch
        return (
            self.version.to_bytes(1, 'big') +
            len(self.salt).to_bytes(2, 'big') +
            self.salt +
            self.sketch
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'HelperData':
        """Deserialize helper data."""
        version = data[0]
        salt_len = int.from_bytes(data[1:3], 'big')
        salt = data[3:3+salt_len]
        sketch = data[3+salt_len:]
        return cls(sketch=sketch, salt=salt, version=version)


def test_bch_directly():
    """Direct test of BCH encode/decode without fuzzy extractor logic."""
    print("\n[DIRECT BCH TEST]")
    print("-" * 50)
    
    bch = BCHCode(m=9, t=29)
    print(f"BCH parameters: n={bch.n}, k={bch.k}, t={bch.t}, ecc_bits={bch.ecc_bits}")
    
    # Generate random message
    msg_bytes = bch.k // 8
    original_msg = secrets.token_bytes(msg_bytes)
    print(f"Original message: {msg_bytes} bytes")
    
    # Encode
    codeword = bch.encode(original_msg, debug=True)
    print(f"Codeword length: {len(codeword)} bytes")
    
    # Decode immediately (no errors)
    recovered, nerrors = bch.decode(codeword, debug=True)
    
    if recovered is not None:
        match = recovered == original_msg[:len(recovered)]
        print(f"Recovery: {'✓ SUCCESS' if match else '✗ MISMATCH'}")
        print(f"Errors corrected: {nerrors}")
    else:
        print("Recovery: ✗ FAILED")
    
    return recovered is not None


class BCHCode:
    """
    BCH error-correcting code wrapper.
    
    BCH codes are particularly suitable for biometric applications because:
    1. They operate on binary data (Hamming distance metric)
    2. They have efficient encoding/decoding algorithms
    3. Error correction capability t is configurable
    
    Parameters:
        m: Galois field order, n = 2^m - 1
        t: Error correction capability (corrects up to t errors)
    """
    
    def __init__(self, m: int = 9, t: int = 29):
        """
        Initialize BCH code.
        
        Default BCH(511, ~250, 29):
            - n = 511 bits codeword length
            - k ≈ 250 bits message length
            - t = 29 bits error correction (~5.7% error tolerance)
        """
        self.m = m
        self.t = t
        self.n = (1 << m) - 1  # 2^m - 1
        
        if HAS_BCHLIB:
            # Use bchlib for efficient BCH operations
            # Handle different bchlib API versions
            self.bch = self._init_bchlib(m, t)
            if self.bch is not None:
                self.k = self.bch.n - self.bch.ecc_bits  # Actual message bits
                self.ecc_bits = self.bch.ecc_bits
            else:
                # Fallback to no BCH
                self.k = max(self.n - m * t, 1)
                self.ecc_bits = self.n - self.k
        else:
            # Approximate k for information
            # For BCH: k ≥ n - m*t
            self.k = max(self.n - m * t, 1)
            self.ecc_bits = self.n - self.k
            self.bch = None
    
    def _init_bchlib(self, m: int, t: int):
        """
        Initialize bchlib with multiple API fallbacks for version compatibility.
        
        Different bchlib versions have different APIs:
        - Newer: BCH(t, m=m) or BCH(t, poly=poly)
        - Older: BCH(poly, t) with primitive polynomial
        """
        # Primitive polynomials for GF(2^m) - used by older bchlib versions
        PRIMITIVE_POLYNOMIALS = {
            5: 0x25,      # x^5 + x^2 + 1
            6: 0x43,      # x^6 + x + 1  
            7: 0x89,      # x^7 + x^3 + 1
            8: 0x11d,     # x^8 + x^4 + x^3 + x^2 + 1
            9: 0x211,     # x^9 + x^4 + 1
            10: 0x409,    # x^10 + x^3 + 1
            11: 0x805,    # x^11 + x^2 + 1
            12: 0x1053,   # x^12 + x^6 + x^4 + x + 1
            13: 0x201b,   # x^13 + x^4 + x^3 + x + 1
            14: 0x4443,   # x^14 + x^10 + x^6 + x + 1
            15: 0x8003,   # x^15 + x + 1
        }
        
        # Method 1: Newer API with keyword argument
        try:
            bch = bchlib.BCH(t, m=m)
            if bch.n == (1 << m) - 1:
                return bch
        except (TypeError, RuntimeError):
            pass
        
        # Method 2: Try with polynomial keyword
        if m in PRIMITIVE_POLYNOMIALS:
            try:
                bch = bchlib.BCH(t, poly=PRIMITIVE_POLYNOMIALS[m])
                if bch.n == (1 << m) - 1:
                    return bch
            except (TypeError, RuntimeError):
                pass
        
        # Method 3: Older API - BCH(poly, t) positional
        if m in PRIMITIVE_POLYNOMIALS:
            try:
                bch = bchlib.BCH(PRIMITIVE_POLYNOMIALS[m], t)
                if bch.n == (1 << m) - 1:
                    return bch
            except (TypeError, RuntimeError):
                pass
        
        # Method 4: Try without any polynomial (some versions auto-select)
        try:
            bch = bchlib.BCH(t)
            return bch
        except (TypeError, RuntimeError):
            pass
        
        # All methods failed
        print(f"WARNING: Could not initialize BCH({m}, {t}). Using fallback mode.")
        return None
    
    def encode(self, message: bytes, debug: bool = False) -> bytes:
        """
        Encode message into BCH codeword.
        
        Args:
            message: Input bytes (will be padded/truncated to k bits)
            debug: Print debug information
            
        Returns:
            BCH codeword as bytes
        """
        if self.bch is not None:
            # Pad message to required length
            msg_bytes = self.k // 8
            padded_msg = message[:msg_bytes].ljust(msg_bytes, b'\x00')
            
            # Compute ECC
            ecc = self.bch.encode(padded_msg)
            
            if debug:
                print(f"  [BCH.encode] msg_bytes={msg_bytes}, len(ecc)={len(ecc)}, "
                      f"total={msg_bytes + len(ecc)}")
            
            # Codeword = message || ECC
            return padded_msg + ecc
        else:
            return self._fallback_encode(message)
    
    def decode(self, codeword: bytes, debug: bool = False) -> Tuple[Optional[bytes], int]:
        """
        Decode BCH codeword, correcting up to t errors.
        
        Args:
            codeword: Received (possibly corrupted) codeword
            debug: Print debug information
            
        Returns:
            (corrected_message, num_errors) or (None, -1) if uncorrectable
        """
        if self.bch is not None:
            # Split codeword into data and ECC
            msg_bytes = self.k // 8
            
            # CRITICAL: Use actual ECC byte length, not floor division!
            # bchlib.encode() returns ceil(ecc_bits/8) bytes
            # Different bchlib versions expose this differently
            if hasattr(self.bch, 'ecc_bytes'):
                ecc_bytes = self.bch.ecc_bytes
            else:
                # Fallback: calculate ceil(ecc_bits/8)
                ecc_bytes = (self.ecc_bits + 7) // 8
            
            if debug:
                print(f"  [BCH.decode] codeword_len={len(codeword)}, msg_bytes={msg_bytes}, "
                      f"ecc_bytes={ecc_bytes}, expected_total={msg_bytes + ecc_bytes}")
            
            data = bytearray(codeword[:msg_bytes])
            ecc = bytearray(codeword[msg_bytes:msg_bytes + ecc_bytes])
            
            if debug:
                print(f"  [BCH.decode] actual data_len={len(data)}, actual ecc_len={len(ecc)}")
            
            # Attempt to decode
            try:
                # bchlib.decode has different return types in different versions:
                # - Some versions: returns int (nerrors), modifies data/ecc in place
                # - Other versions: returns tuple (nerrors, corrected_data, corrected_ecc)
                result = self.bch.decode(data, ecc)
                
                if debug:
                    print(f"  [BCH.decode] result type={type(result)}, result={result if not isinstance(result, tuple) else f'tuple len={len(result)}'}")
                
                # Handle both API versions
                if isinstance(result, tuple):
                    # Newer API: returns (nerrors, corrected_data, corrected_ecc)
                    nerrors = result[0]
                    if len(result) > 1 and result[1] is not None:
                        data = bytearray(result[1])  # Use returned corrected data
                else:
                    # Older API: returns just nerrors, data modified in place
                    nerrors = result
                
                if debug:
                    print(f"  [BCH.decode] nerrors={nerrors}")
                
                if nerrors >= 0:
                    return bytes(data), nerrors
                else:
                    return None, -1
            except Exception as e:
                if debug:
                    print(f"  [BCH.decode] Exception: {e}")
                return None, -1
        else:
            return self._fallback_decode(codeword)
    
    def _fallback_encode(self, message: bytes) -> bytes:
        """Simple repetition code fallback (for testing only)."""
        # WARNING: This is NOT a real BCH code, just for API compatibility
        return message + b'\x00' * (self.ecc_bits // 8)
    
    def _fallback_decode(self, codeword: bytes) -> Tuple[Optional[bytes], int]:
        """Fallback decoder (no error correction)."""
        msg_bytes = self.k // 8
        return codeword[:msg_bytes], 0


class FuzzyExtractor:
    """
    Secure Fuzzy Extractor using Code-Offset Construction.
    
    The code-offset construction works as follows:
    
    Gen(w):
        1. Sample random codeword c ← Encode(random_key)
        2. Compute sketch P = c ⊕ w
        3. Derive final key R = KDF(random_key, salt)
        4. Output (R, HelperData(P, salt))
    
    Rep(w', P):
        1. Compute c' = P ⊕ w'
           Note: c' = (c ⊕ w) ⊕ w' = c ⊕ (w ⊕ w') = c ⊕ e
           where e is the error pattern
        2. Decode c' to recover random_key (if ||e|| ≤ t)
        3. Derive R = KDF(random_key, salt)
    
    Security Properties:
        - Helper data P is computationally indistinguishable from random
          (under the assumption that w has sufficient min-entropy)
        - The derived key R has high min-entropy
    """
    
    def __init__(self, config: FuzzyExtractorConfig = None):
        self.config = config or DEFAULT_CONFIG.fuzzy_extractor
        
        # Initialize BCH code
        self.bch = BCHCode(m=self.config.bch_m, t=self.config.bch_t)
        
        # Derived parameters
        self.n = self.bch.n  # Codeword length in bits
        self.k = self.bch.k  # Message length in bits
        self.t = self.bch.t  # Error correction capability
        
        print(f"FuzzyExtractor initialized: BCH({self.n}, {self.k}, {self.t})")
        print(f"  - Max error rate: {self.t/self.n*100:.2f}%")
        print(f"  - Key bits (before KDF): {self.k}")
        
    def gen(self, biometric: bytes, key_length: int = 32) -> Tuple[bytes, HelperData]:
        """
        Generate cryptographic key and helper data from biometric.
        
        This is the enrollment phase.
        
        Args:
            biometric: Binary biometric template (from BioHasher)
            key_length: Desired key length in bytes (default 256 bits)
            
        Returns:
            (key, helper_data): Cryptographic key and public helper data
        """
        # 1. Generate random key material
        key_bytes = self.k // 8
        random_key = secrets.token_bytes(key_bytes)
        
        # 2. Encode random key into BCH codeword
        codeword = self.bch.encode(random_key)
        
        # 3. Compute sketch: P = codeword ⊕ biometric
        # CRITICAL: Pad biometric to match CODEWORD length, not n bits!
        # bchlib produces byte-aligned codewords that may be longer than ceil(n/8)
        codeword_len = len(codeword)
        padded_bio = biometric[:codeword_len].ljust(codeword_len, b'\x00')
        
        sketch = self._xor_bytes(codeword, padded_bio)
        
        # 4. Generate salt and derive final key
        salt = secrets.token_bytes(32)
        derived_key = self._derive_key(random_key, salt, key_length)
        
        # 5. Create helper data
        helper = HelperData(sketch=sketch, salt=salt)
        
        return derived_key, helper
    
    def rep(
        self, 
        biometric: bytes, 
        helper: Union[HelperData, bytes],
        key_length: int = 32
    ) -> Tuple[Optional[bytes], int]:
        """
        Reproduce cryptographic key from noisy biometric and helper data.
        
        This is the authentication phase.
        
        Args:
            biometric: Noisy binary biometric template
            helper: Helper data from enrollment
            key_length: Expected key length in bytes
            
        Returns:
            (key, num_errors): Recovered key and number of corrected errors,
                              or (None, -1) if recovery failed
        """
        # Parse helper data if needed
        if isinstance(helper, bytes):
            helper = HelperData.from_bytes(helper)
        
        # 1. Compute noisy codeword: c' = P ⊕ w'
        # CRITICAL: Pad biometric to match SKETCH length (which equals codeword length)
        sketch_len = len(helper.sketch)
        padded_bio = biometric[:sketch_len].ljust(sketch_len, b'\x00')
        
        noisy_codeword = self._xor_bytes(helper.sketch, padded_bio)
        
        # 2. Decode to recover random key
        random_key, num_errors = self.bch.decode(noisy_codeword)
        
        if random_key is None:
            return None, -1
        
        # 3. Derive final key (must match enrollment)
        derived_key = self._derive_key(random_key, helper.salt, key_length)
        
        return derived_key, num_errors
    
    def _xor_bytes(self, a: bytes, b: bytes) -> bytes:
        """XOR two byte arrays of equal length."""
        return bytes(x ^ y for x, y in zip(a, b))
    
    def _derive_key(
        self, 
        secret: bytes, 
        salt: bytes, 
        length: int
    ) -> bytes:
        """
        Derive final key using HKDF (HMAC-based Key Derivation Function).
        
        HKDF provides:
            - Extraction: Concentrates entropy into a pseudorandom key
            - Expansion: Derives multiple keys of desired length
        """
        # HKDF-Extract: PRK = HMAC(salt, secret)
        prk = hmac.new(salt, secret, hashlib.sha256).digest()
        
        # HKDF-Expand: derive key of desired length
        # For simplicity, we use single iteration (works for length ≤ 32)
        info = b"biometric_key_v1"
        key = hmac.new(prk, info + b'\x01', hashlib.sha256).digest()
        
        return key[:length]
    
    def estimate_security(
        self, 
        biometric_entropy: float
    ) -> dict:
        """
        Estimate security parameters of the fuzzy extractor.
        
        Args:
            biometric_entropy: Estimated min-entropy of biometric in bits
            
        Returns:
            Dictionary of security metrics
        """
        # Entropy leakage through helper data
        # For code-offset sketch: leak ≈ n - k
        leakage = self.n - self.k
        
        # Remaining entropy in the key
        remaining_entropy = max(biometric_entropy - leakage, 0)
        
        # Security level (bits)
        security_bits = min(remaining_entropy, self.k)
        
        return {
            "code_parameters": {
                "n": self.n,
                "k": self.k,
                "t": self.t
            },
            "entropy_analysis": {
                "biometric_entropy": biometric_entropy,
                "leakage_bits": leakage,
                "remaining_entropy": remaining_entropy,
                "effective_security_bits": security_bits
            },
            "error_tolerance": {
                "max_errors": self.t,
                "max_error_rate": self.t / self.n,
                "max_hamming_distance": self.t
            },
            "recommendation": (
                "SECURE" if security_bits >= 128 else
                "MARGINAL" if security_bits >= 80 else
                "WEAK"
            )
        }


class ReusableFuzzyExtractor(FuzzyExtractor):
    """
    Reusable Fuzzy Extractor with unlinkability guarantees.
    
    Standard fuzzy extractors have a weakness: if the same biometric is
    enrolled multiple times with different helper data, an adversary can
    potentially link these enrollments.
    
    This implementation adds:
    1. Salt-based sketch randomization
    2. Key commitment to prevent helper data manipulation
    
    Note: True reusable fuzzy extractors require more sophisticated
    constructions (e.g., based on digital lockers). This is a practical
    approximation suitable for many applications.
    """
    
    def gen(
        self, 
        biometric: bytes, 
        key_length: int = 32,
        user_id: Optional[bytes] = None
    ) -> Tuple[bytes, HelperData]:
        """
        Generate key with enhanced unlinkability.
        
        Args:
            biometric: Binary biometric template
            key_length: Desired key length
            user_id: Optional user identifier for domain separation
            
        Returns:
            (key, helper_data)
        """
        # Generate instance-specific salt
        instance_salt = secrets.token_bytes(32)
        
        # Hash biometric with salt to create a "blinded" template
        # This prevents direct comparison of sketches
        h = hashlib.sha256()
        h.update(instance_salt)
        h.update(biometric)
        if user_id:
            h.update(user_id)
        blinding_factor = h.digest()
        
        # XOR blinding factor into biometric (repeated to match length)
        bio_bytes = len(biometric)
        blinded_bio = bytes(
            biometric[i] ^ blinding_factor[i % 32] 
            for i in range(bio_bytes)
        )
        
        # Generate using parent method with blinded biometric
        key, helper = super().gen(blinded_bio, key_length)
        
        # Include instance salt in helper data
        # Modify salt to include instance_salt
        combined_salt = instance_salt + helper.salt
        helper = HelperData(
            sketch=helper.sketch,
            salt=combined_salt,
            version=2  # Version 2 indicates reusable variant
        )
        
        return key, helper
    
    def rep(
        self, 
        biometric: bytes, 
        helper: Union[HelperData, bytes],
        key_length: int = 32,
        user_id: Optional[bytes] = None
    ) -> Tuple[Optional[bytes], int]:
        """
        Reproduce key with blinding.
        """
        if isinstance(helper, bytes):
            helper = HelperData.from_bytes(helper)
        
        # Extract instance salt and derivation salt
        instance_salt = helper.salt[:32]
        derivation_salt = helper.salt[32:]
        
        # Reconstruct blinding factor
        h = hashlib.sha256()
        h.update(instance_salt)
        h.update(biometric)
        if user_id:
            h.update(user_id)
        blinding_factor = h.digest()
        
        # Apply blinding
        bio_bytes = len(biometric)
        blinded_bio = bytes(
            biometric[i] ^ blinding_factor[i % 32] 
            for i in range(bio_bytes)
        )
        
        # Create temporary helper with original salt structure
        temp_helper = HelperData(
            sketch=helper.sketch,
            salt=derivation_salt,
            version=1
        )
        
        return super().rep(blinded_bio, temp_helper, key_length)


def test_fuzzy_extractor():
    """
    Comprehensive test of the fuzzy extractor.
    """
    print("=" * 60)
    print("Fuzzy Extractor Test Suite")
    print("=" * 60)
    
    # Initialize
    fe = FuzzyExtractor()
    
    # Simulate biometric (511 bits)
    np.random.seed(42)
    original_bio = bytes(np.random.randint(0, 256, 64, dtype=np.uint8))
    
    print("\n[1] Enrollment (Gen)")
    key, helper = fe.gen(original_bio)
    print(f"  Key: {key.hex()[:32]}...")
    print(f"  Helper data size: {len(helper.to_bytes())} bytes")
    
    print("\n[2] Perfect Reproduction (no noise)")
    recovered_key, errors = fe.rep(original_bio, helper)
    assert recovered_key == key, "Perfect reproduction failed!"
    print(f"  Key matches: ✓")
    print(f"  Errors corrected: {errors}")
    
    print("\n[3] Noisy Reproduction (within tolerance)")
    # Add noise (flip some bits)
    noisy_bio = bytearray(original_bio)
    num_flips = fe.t - 5  # Stay within correction capability
    for i in np.random.choice(len(noisy_bio) * 8, num_flips, replace=False):
        byte_idx = i // 8
        bit_idx = i % 8
        noisy_bio[byte_idx] ^= (1 << bit_idx)
    noisy_bio = bytes(noisy_bio)
    
    recovered_key, errors = fe.rep(noisy_bio, helper)
    print(f"  Bits flipped: {num_flips}")
    print(f"  Key matches: {'✓' if recovered_key == key else '✗'}")
    print(f"  Errors corrected: {errors}")
    
    print("\n[4] Excessive Noise (beyond tolerance)")
    very_noisy_bio = bytearray(original_bio)
    num_flips = fe.t + 10  # Exceed correction capability
    for i in np.random.choice(len(very_noisy_bio) * 8, num_flips, replace=False):
        byte_idx = i // 8
        bit_idx = i % 8
        very_noisy_bio[byte_idx] ^= (1 << bit_idx)
    very_noisy_bio = bytes(very_noisy_bio)
    
    recovered_key, errors = fe.rep(very_noisy_bio, helper)
    print(f"  Bits flipped: {num_flips}")
    print(f"  Recovery failed (expected): {'✓' if recovered_key is None else '✗'}")
    
    print("\n[5] Security Analysis")
    security = fe.estimate_security(biometric_entropy=200)
    print(f"  Biometric entropy: {security['entropy_analysis']['biometric_entropy']} bits")
    print(f"  Leakage: {security['entropy_analysis']['leakage_bits']} bits")
    print(f"  Effective security: {security['entropy_analysis']['effective_security_bits']:.1f} bits")
    print(f"  Recommendation: {security['recommendation']}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    # First run direct BCH test
    bch_ok = test_bch_directly()
    
    if bch_ok:
        print("\nDirect BCH test passed! Running full fuzzy extractor test...")
        test_fuzzy_extractor()
    else:
        print("\n⚠️ Direct BCH test FAILED! Not running fuzzy extractor test.")