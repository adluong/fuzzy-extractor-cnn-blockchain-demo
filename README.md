# CNN + Fuzzy Extractor + Blockchain Biometric Authentication

- **Important note: This project is created with the help of Claude and is for testing purpose only, DO NOT use it in product** - adluong.

> Biometric authentication using FaceNet, fuzzy extractors (including post-quantum LWE), and Ethereum smart contracts.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

```bash
# Install dependencies
pip install torch torchvision facenet-pytorch bchlib numpy

# Run benchmark (binary pipeline)
python main.py --mode benchmark

# Run benchmark (True LWE - post-quantum)
python main_true_lwe.py --mode benchmark
```

## Features

| Feature | Description |
|---------|-------------|
| **FaceNet CNN** | Pretrained on VGGFace2, 512-D embeddings |
| **BCH Fuzzy Extractor** | BCH(511, 268, 29), information-theoretic |
| **Repetition-Code FE** | Code-offset scheme, 5.7% error tolerance |
| **True LWE FE** | **Post-quantum secure**, continuous embeddings |
| **Improved BioHash** | Reliable bit selection, 0.7% intra-class variation |
| **Blockchain Auth** | Ethereum smart contract (mock or real) |

## Two Pipelines

### Binary Pipeline (main.py)
```
Face → FaceNet → Embedding → BioHash → Binary Code → BCH/Rep FE → Key
```
- Uses BioHash for binary conversion
- Error model: Hamming distance (bit flips)
- FE options: BCH or Repetition-Code

### Continuous Pipeline (main_true_lwe.py) — NEW
```
Face → FaceNet → Embedding → True LWE FE → Key
```
- Works directly on continuous embeddings
- Error model: Euclidean distance
- **Post-quantum secure** via LWE hardness

## Benchmark Results

### Binary Pipeline (Repetition-Code FE)
```
python main.py --mode benchmark

  ✓ Bit flips  0: FRR =   0.0%
  ✓ Bit flips 29: FRR =  14.0%  ← Threshold (5.7%)
  ✗ Bit flips 30: FRR =  92.0%  ← Correctly rejects
```

### Continuous Pipeline (True LWE FE)
```
python main_true_lwe.py --mode benchmark

  ✓ Noise   0%: FRR =   0.0%
  ✓ Noise   8%: FRR =   0.0%
  ✓ Noise  10%: FRR =  20.0%  ← Near threshold
  ✗ Noise  15%: FRR = 100.0%  ← Correctly rejects

  FAR: 0.000%
  Post-Quantum: Yes
  Security: ~797 bits
```

## Why True LWE Works: Field Elements vs Bits

### The Problem with Binary BioHash + LWE

LWE operates on **field elements** in ℤ_q (integers mod q), while BioHash produces **binary bits**:

```
LWE error model:     c = A·b + e + M  where e ~ Gaussian(σ=1.4)
                     Decoding works if |error| < q/4

Binary bit flip:     If bit b encoded as field element:
                     b=0 → 0x00080000, b=1 → 0x00180000
                     
                     A single bit flip causes Δ = 1,048,576
                     This EXCEEDS LWE's Gaussian error tolerance!
```

**Result:** Binary BioHash cannot use LWE for error correction.

### The Solution: Continuous Quantization

True LWE FE works directly on continuous embeddings:

```python
def quantize(embedding):
    """Map continuous embedding to field elements."""
    # Normalize to unit vector
    normalized = embedding / ‖embedding‖
    
    # Map [-1, 1] → [q/4, 3q/4] with margin for LWE error
    margin = q // 4
    quantized = (normalized + 1) / 2 * (q - 2*margin) + margin
    
    return quantized % q
```

**Key insight:** Small embedding changes → small field element changes

```
┌────────────────────────────────────────────────────────────────────────┐
│ Embedding space:     [-1 ────────── 0 ────────── +1]                   │
│                             ↓ quantize ↓                               │
│ Field element space: [0 ── q/4 ──── q/2 ──── 3q/4 ── q]               │
│                          margin           margin                       │
│                                                                        │
│ • Small noise (Δ=0.1) → Small field change (~840K)                    │
│ • LWE Gaussian error σ=1.4 → Negligible vs margin (4M)                │
│ • Euclidean distance preserved through quantization                    │
└────────────────────────────────────────────────────────────────────────┘
```

### Comparison

| Aspect | Binary (BioHash) | Continuous (True LWE) |
|--------|-----------------|----------------------|
| Input | 511 bits | 512-D float vector |
| Error model | Hamming (bit flips) | Euclidean (magnitude) |
| Error tolerance | 29 bits (5.7%) | ~10% L2 distance |
| LWE compatible | ❌ No | ✅ Yes |
| Post-quantum | ❌ | ✅ |

## Project Structure (Essential Files)

```
biometric_auth/
├── main.py                      # Binary pipeline entry point
├── main_true_lwe.py             # Continuous pipeline (post-quantum)
├── config.py                    # Configuration
├── model.py                     # FaceNet encoder
├── biohashing.py                # Standard BioHasher
├── biohashing_improved.py       # Improved BioHasher
├── fuzzy_extractor.py           # BCH-based FE
├── fuzzy_extractor_lwe.py       # Repetition-code FE
├── fuzzy_extractor_true_lwe.py  # True LWE FE (post-quantum)
├── blockchain_client.py         # Ethereum client
└── contracts/
    └── BiometricAuth.sol        # Smart contract
```

## Usage

### Demo Mode
```bash
# Binary pipeline
python main.py --mode demo

# True LWE pipeline (post-quantum)
python main_true_lwe.py --mode demo
```

### Benchmark Mode
```bash
# Binary pipeline benchmark
python main.py --mode benchmark

# True LWE benchmark
python main_true_lwe.py --mode benchmark
```

## Fuzzy Extractor Selection

### Option 1: Binary Pipeline (Hamming distance errors)
```python
# Uses BioHash → BCH or Repetition-Code FE
from main import BiometricPipeline
pipeline = BiometricPipeline()
```

### Option 2: Continuous Pipeline (Euclidean distance errors)
```python
# Uses embeddings directly → True LWE FE (post-quantum)
from main_true_lwe import TrueLWEBiometricPipeline
pipeline = TrueLWEBiometricPipeline()
```

### When to Use Each

| Use Case | Recommended |
|----------|-------------|
| Need post-quantum security | `main_true_lwe.py` |
| Working with binary templates | `main.py` |
| Minimal helper data | `main.py` (BCH) |
| Direct embedding processing | `main_true_lwe.py` |

## API

### Binary Pipeline
```python
from main import BiometricPipeline

pipeline = BiometricPipeline(use_mock_blockchain=True)
key, helper, keypair = pipeline.enroll(image)
success, errors = pipeline.authenticate(image, helper)
```

### True LWE Pipeline (Post-Quantum)
```python
from main_true_lwe import TrueLWEBiometricPipeline

pipeline = TrueLWEBiometricPipeline()
key, helper, embedding = pipeline.enroll(image)
success, distance, recovered = pipeline.authenticate(image, helper, key)
```

## Documentation

- **[PROJECT_REPORT5.md](PROJECT_REPORT5.md)** — Complete technical documentation (v5)
- **[PROJECT_REPORT4.md](PROJECT_REPORT4.md)** — Binary pipeline documentation (v4)

## Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (recommended for GPU acceleration)
- facenet-pytorch
- bchlib

```bash
pip install -r requirements.txt
```

## License

MIT License - see LICENSE file.

## References

1. Dodis et al., "Fuzzy Extractors: How to Generate Strong Keys from Biometrics", SIAM 2008
2. Apon et al., "Efficient, Reusable Fuzzy Extractors from LWE", 2017
3. Regev, "On Lattices, Learning with Errors", STOC 2005
4. Schroff et al., "FaceNet: A Unified Embedding for Face Recognition", CVPR 2015