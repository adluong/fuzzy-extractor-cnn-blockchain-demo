# CNN + Fuzzy Extractor + Blockchain Biometric Authentication

> Biometric authentication using FaceNet, fuzzy extractors, and Ethereum smart contracts.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quick Start

```bash
# Install dependencies
pip install torch torchvision facenet-pytorch bchlib numpy

# Run benchmark
python main.py --mode benchmark

# Run demo
python main.py --mode demo
```

## Features

| Feature | Description |
|---------|-------------|
| **FaceNet CNN** | Pretrained on VGGFace2, 512-D embeddings |
| **BCH Fuzzy Extractor** | BCH(511, 268, 29), information-theoretic security |
| **Repetition-Code FE** | Code-offset with repetition codes, 5.7% error tolerance |
| **Improved BioHash** | Reliable bit selection, 0.7% intra-class variation |
| **Blockchain Auth** | Ethereum smart contract (mock or real) |

## Benchmark Results

```
python main.py --mode benchmark
```

### Repetition-Code FE Error Correction
```
  ✓ Bit flips  0: FRR =   0.0%
  ✓ Bit flips 29: FRR =  14.0%  ← Threshold
  ✗ Bit flips 30: FRR =  92.0%  ← Correctly rejects
```

### BioHash Comparison
```
Metric                    Standard        Improved
-------------------------------------------------
Genuine Hamming           137 bits (27%)  1.4 bits (0.7%)
FRR                       100%            0%
FAR                       0%              0%
```

### Performance
```
Enrollment:     25.88 ms
Authentication: 25.34 ms
```

## Project Structure (Essential Files)

```
biometric_auth/
├── main.py                  # Entry point
├── config.py                # Configuration
├── model.py                 # FaceNet encoder
├── biohashing.py            # Standard BioHasher
├── biohashing_improved.py   # Improved BioHasher
├── fuzzy_extractor.py       # BCH-based FE
├── fuzzy_extractor_lwe.py   # Repetition-code FE (code-offset scheme)
├── blockchain_client.py     # Ethereum client
└── contracts/
    └── BiometricAuth.sol    # Smart contract
```

## Usage

### Demo Mode
```bash
python main.py --mode demo
```
Runs end-to-end demonstration: enrollment → authentication → impostor rejection.

### Benchmark Mode
```bash
python main.py --mode benchmark
```
Runs comprehensive 5-section evaluation:
1. Fuzzy extractor error correction unit test
2. Standard BioHash evaluation
3. Improved BioHash evaluation
4. Performance benchmark
5. Blockchain cost analysis

### Evaluation Scripts
```bash
# LFW real face evaluation
python evaluate_improved.py --mode lfw

# Blockchain gas costs
python evaluate_blockchain.py
```

## Fuzzy Extractor Selection

The system auto-selects the repetition-code FE by default:

```python
# In main.py - automatic selection
try:
    from fuzzy_extractor_lwe import LWEFuzzyExtractor as FuzzyExtractor
    # Repetition-code based (code-offset scheme)
except ImportError:
    from fuzzy_extractor import FuzzyExtractor
    # BCH-based
```

To force BCH:
```bash
mv fuzzy_extractor_lwe.py fuzzy_extractor_lwe.py.bak
python main.py --mode benchmark
```

## API

```python
from main import BiometricPipeline

# Initialize
pipeline = BiometricPipeline(use_mock_blockchain=True)

# Enrollment
image = load_face_image("face.jpg")
secret_key, helper_data, keypair = pipeline.enroll(image)

# Authentication
success, errors = pipeline.authenticate(image, helper_data)
```

## Blockchain Networks

| Network | Registration | Auth | Recommended |
|---------|--------------|------|-------------|
| Ethereum | $13.61 | $10.12 | ❌ Expensive |
| Polygon | $0.008 | $0.006 | ✅ Production |
| Arbitrum | $0.068 | $0.051 | ✅ Production |
| Local | FREE | FREE | ✅ Development |

## Documentation

- **[PROJECT_REPORT4.md](PROJECT_REPORT4.md)** — Complete technical documentation
- **[contracts/BiometricAuth.sol](contracts/BiometricAuth.sol)** — Smart contract source

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
2. Schroff et al., "FaceNet: A Unified Embedding for Face Recognition", CVPR 2015