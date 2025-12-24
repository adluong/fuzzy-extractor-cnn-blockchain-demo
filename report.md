# CNN + Fuzzy Extractor + Blockchain Biometric Authentication

## Project Report

**Date:** December 24, 2025  
**Version:** 1.0.0  
**Status:** Development / Testing Phase

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture](#2-system-architecture)
3. [Project Structure](#3-project-structure)
4. [Component Details](#4-component-details)
5. [Test Results](#5-test-results)
6. [Issue Analysis](#6-issue-analysis)
7. [Recommendations](#7-recommendations)
8. [Usage Guide](#8-usage-guide)

---

## 1. Executive Summary

This project implements a **biometric authentication system** that combines:

- **Deep Learning (CNN):** ResNet-50 with ArcFace loss for biometric feature extraction
- **Fuzzy Extractors:** BCH error-correcting codes for noise-tolerant key generation
- **Blockchain:** Ethereum smart contract for decentralized authentication

### Key Features

| Feature | Description |
|---------|-------------|
| **Cancelable Biometrics** | User-specific tokens enable template revocation |
| **Noise Tolerance** | BCH(511, 268, 29) corrects up to 29 bit errors (5.7%) |
| **Decentralized** | No trusted third party; helper data stored on-chain |
| **Privacy-Preserving** | Raw biometrics never leave the device |

### Current Status

⚠️ **Critical Issue Detected:** Authentication fails even with identical biometric inputs (FRR = 100%). See [Section 6](#6-issue-analysis) for root cause analysis.

---

## 2. System Architecture

### 2.1 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ENROLLMENT PHASE                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Biometric    CNN Encoder     BioHasher      Fuzzy Extractor        │
│    Image   ─────────────► ──────────────► ─────────────────►        │
│  (112×112)   512-D embed    511-bit binary    Gen(w) → (R, P)       │
│                                                    │                 │
│                                                    ▼                 │
│                                              ┌─────────────┐         │
│                                              │ Key R       │         │
│                                              │ Helper P    │         │
│                                              └─────────────┘         │
│                                                    │                 │
│                               ┌────────────────────┴───────┐         │
│                               ▼                            ▼         │
│                        ECDSA Keypair              Store on Chain     │
│                        derivation                 (helper data)      │
│                               │                            │         │
│                               ▼                            ▼         │
│                        Public Key Hash ──────────► Blockchain        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                      AUTHENTICATION PHASE                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Biometric'   CNN Encoder     BioHasher      Fuzzy Extractor        │
│    Image   ─────────────► ──────────────► ─────────────────►        │
│  (noisy)     512-D embed    511-bit binary   Rep(w', P) → R'        │
│                                                    │                 │
│                                   ┌────────────────┘                 │
│                                   ▼                                  │
│                            ┌─────────────┐                           │
│                            │ R' == R ?   │                           │
│                            └─────────────┘                           │
│                                   │                                  │
│                     ┌─────────────┴─────────────┐                    │
│                     ▼                           ▼                    │
│               ✓ Success                    ✗ Failure                 │
│            (sign challenge)            (too many errors)             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Security Model

| Property | Guarantee |
|----------|-----------|
| **Irreversibility** | Cannot recover biometric from helper data P |
| **Unlinkability** | Different enrollments produce unlinkable templates |
| **Replay Prevention** | Challenge-response with nonce + expiry |
| **Key Secrecy** | Key R has min-entropy ≥ k - leakage bits |

---

## 3. Project Structure

```
.
├── BiometricAuth.sol          # Ethereum smart contract
├── README.md                  # Project documentation
├── biohashing.py              # Random projection binarization
├── blockchain_client.py       # Web3.py contract interface
├── config.py                  # Centralized configuration
├── evaluate.py                # FAR/FRR evaluation suite
├── feature_extractor_lite.py  # Numpy-only feature extractor
├── fuzzy_extractor.py         # BCH-based fuzzy extractor
├── main.py                    # Full pipeline (requires PyTorch)
├── main_lite.py               # Lightweight pipeline (numpy only)
├── model.py                   # ResNet-50 + ArcFace CNN
├── requirements.txt           # Full dependencies
├── requirements_lite.txt      # Lightweight dependencies
└── train.py                   # CNN training script
```

### 3.1 Dependency Comparison

| Component | Full Version (`main.py`) | Lite Version (`main_lite.py`) |
|-----------|--------------------------|-------------------------------|
| Feature Extractor | ResNet-50 + ArcFace | Gabor + LBP / Hash-based |
| Dependencies | PyTorch, torchvision | numpy, bchlib, cryptography |
| Model Size | ~95 MB | 0 (no model file) |
| GPU Support | Yes | No |
| Production Ready | Yes (after training) | Testing only |

---

## 4. Component Details

### 4.1 CNN Feature Extractor (`model.py`)

**Architecture:**
```
Input (3×112×112) 
    │
    ▼
ResNet-50 (pretrained ImageNet)
    │
    ▼
Global Average Pooling
    │
    ▼
FC Layer (2048 → 512)
    │
    ▼
BatchNorm + L2 Normalize
    │
    ▼
Output: 512-D unit vector
```

**ArcFace Loss:**
$$L = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos\theta_j}}$$

Where:
- $s = 64$ (scale factor)
- $m = 0.5$ (angular margin, ~28.6°)

### 4.2 BioHashing (`biohashing.py`)

**Algorithm:**
1. Generate orthonormal random matrix $\mathbf{R} \in \mathbb{R}^{512 \times 511}$ (Gram-Schmidt)
2. Project embedding: $\mathbf{p} = \mathbf{e} \cdot \mathbf{R}$
3. Binarize: $b_i = \mathbb{1}[p_i > 0]$

**Properties:**
- Input: 512-D float embedding
- Output: 511-bit binary string
- User token enables cancelability

### 4.3 Fuzzy Extractor (`fuzzy_extractor.py`)

**Code-Offset Construction:**

| Operation | Formula |
|-----------|---------|
| **Gen(w)** | $R \leftarrow \text{Random}$, $C \leftarrow \text{Encode}(R)$, $P \leftarrow C \oplus w$ |
| **Rep(w', P)** | $C' \leftarrow P \oplus w'$, $R \leftarrow \text{Decode}(C')$ |

**BCH Parameters:**

| Parameter | Value | Description |
|-----------|-------|-------------|
| $n$ | 511 | Codeword length (bits) |
| $k$ | 268 | Message length (bits) |
| $t$ | 29 | Error correction capability |
| Max error rate | 5.68% | $t/n = 29/511$ |
| Entropy leakage | 243 bits | $n - k$ |

### 4.4 Blockchain Contract (`BiometricAuth.sol`)

**Key Functions:**

```solidity
function register(bytes helperData, bytes32 publicKeyHash) external;
function requestChallenge() external returns (bytes32 nonce, uint256 expiry);
function authenticate(bytes signature) external returns (bool success);
function getHelperData(address user) external view returns (bytes);
```

**Gas Costs (estimated):**
- Registration: ~150,000 gas
- Authentication: ~80,000 gas

---

## 5. Test Results

### 5.1 Full Version (`main.py --mode demo`)

```
======================================================================
CNN + FUZZY EXTRACTOR + BLOCKCHAIN AUTHENTICATION DEMO
======================================================================
Pipeline Configuration:
  - BCH Parameters: (511, 268, 29)
  - Error tolerance: 5.7% (29 bits)
  
[ENROLLMENT]
  ✓ Secret key generated: a660087757f50ab7...
  ✓ Helper data size: 99 bytes
  ✓ Derived address: 0x8822B6766a4d2f45...
  ✓ Enrollment time: 99.42 ms
  ✓ Blockchain registration: SUCCESS

[AUTHENTICATION - Same Biometric]
  ✗ Result: FAILED
  ✗ Errors corrected: -1

[AUTHENTICATION - Noisy Biometric (5%)]
  ✗ Result: FAILED
  ✗ Errors corrected: -1

[AUTHENTICATION - Impostor]
  ✓ Result: REJECTED (correct behavior)

[SECURITY ANALYSIS]
  ⚠️ Effective security: 0 bits
  ⚠️ Security level: WEAK
```

### 5.2 Full Version (`main.py --mode benchmark`)

```
======================================================================
SECURITY BENCHMARK SUITE
======================================================================
Configuration:
  - Simulated users: 50
  - Auth attempts per noise level: 100
  - Noise levels: [0%, 5%, 10%, 15%, 20%, 25%]

[GENUINE AUTHENTICATION]
┌──────────┬────────┬─────────────────────┐
│ Noise    │ FRR    │ Avg Errors Corrected│
├──────────┼────────┼─────────────────────┤
│ 0%       │ 100.0% │ 0.0                 │
│ 5%       │ 100.0% │ 0.0                 │
│ 10%      │ 100.0% │ 0.0                 │
│ 15%      │ 100.0% │ 0.0                 │
│ 20%      │ 100.0% │ 0.0                 │
│ 25%      │ 100.0% │ 0.0                 │
└──────────┴────────┴─────────────────────┘

[IMPOSTOR DETECTION]
  ✓ False Acceptance Rate (FAR): 0.000%
  ✓ False accepts: 0/100

[PERFORMANCE]
  - Enrollment: 25.74 ± 1.74 ms
  - Authentication: 25.33 ± 2.01 ms
```

### 5.3 Lightweight Version (`main_lite.py`)

```
============================================================
Biometric Authentication Demo (Lightweight Version)
============================================================
BCH(511, 134, 29) initialized

[1] ENROLLMENT
  ✓ User: alice
  ✓ Address: 0x297dAB2640794A53...
  ✓ Helper data: 81 bytes
  ✓ Time: 741.94 ms

[2] AUTHENTICATION (same image)
  ✗ Success: False

[3] AUTHENTICATION (noisy 10%)
  ✗ Hamming distance: 275 bits (>> 29 correctable)
  ✗ Success: False

[4] AUTHENTICATION (impostor)
  ✓ Correctly rejected (259 bit distance)

[5] AUTHENTICATION (with challenge)
  ✗ Success: False
```

---

## 6. Issue Analysis

### 6.1 Root Cause: FRR = 100%

**Problem:** Authentication fails even when using the **exact same biometric image** that was used for enrollment.

**Diagnosis:**

| Check | Expected | Actual | Status |
|-------|----------|--------|--------|
| Same image → Same embedding | ✓ | ✗ Different each time | ❌ **BUG** |
| Same embedding → Same biohash | ✓ | Depends on embedding | — |
| Hamming distance (same user) | < 29 bits | 275 bits | ❌ **Too high** |

**Root Cause Analysis:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    IDENTIFIED ISSUES                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. UNTRAINED CNN MODEL                                          │
│     ─────────────────────                                        │
│     The ResNet-50 uses random/ImageNet weights, not trained     │
│     on biometric data. This causes:                              │
│     • High intra-class variance (same person → different embed) │
│     • Low inter-class variance (different people → similar)     │
│                                                                  │
│  2. NON-DETERMINISTIC INFERENCE                                  │
│     ────────────────────────────                                 │
│     Even with the same input, the CNN may produce different     │
│     outputs due to:                                              │
│     • BatchNorm running statistics                               │
│     • Dropout (if not in eval mode)                              │
│     • Floating-point non-determinism                             │
│                                                                  │
│  3. BIOHASH AMPLIFICATION                                        │
│     ─────────────────────                                        │
│     Small embedding differences get amplified to large          │
│     Hamming distances after binarization:                        │
│     • 1% embedding change → potentially 50% bit flips           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Security Analysis Issue

The reported "Effective security: 0 bits" is a consequence of:

$$\text{Effective Security} = \text{Biometric Entropy} - \text{Leakage}$$

With an untrained CNN:
- Biometric Entropy ≈ 0 (embeddings are essentially random)
- Leakage = 243 bits (from BCH helper data)
- **Effective Security = 0 - 243 = 0 bits** (clamped to 0)

---

## 7. Recommendations

### 7.1 Immediate Fixes

#### Fix 1: Enable Deterministic Mode

```python
# In model.py, add to BiometricEncoder.__init__():
def __init__(self, config):
    super().__init__()
    # ... existing code ...
    
    # Enable deterministic mode
    self.eval()  # Disable dropout, use running stats for BatchNorm
    
def forward(self, x):
    self.eval()  # Ensure eval mode
    with torch.no_grad():
        return self._forward_impl(x)
```

#### Fix 2: Train the CNN

```bash
# Download CASIA-WebFace or FVC2004 dataset
python train.py --dataset casia --data_dir ./data/CASIA-WebFace \
    --batch_size 64 --num_epochs 50 --lr 0.001
```

#### Fix 3: Use Pre-trained Face Recognition Model

```python
# Replace model.py with facenet-pytorch
from facenet_pytorch import InceptionResnetV1

encoder = InceptionResnetV1(pretrained='vggface2').eval()
```

### 7.2 Architecture Improvements

| Improvement | Description | Priority |
|-------------|-------------|----------|
| **Trained CNN** | Use ArcFace-trained model on face/fingerprint data | 🔴 Critical |
| **Deterministic inference** | Ensure same input → same output | 🔴 Critical |
| **Adaptive thresholds** | Learn per-user binarization thresholds | 🟡 Medium |
| **Multi-sample enrollment** | Enroll with N images, average embeddings | 🟡 Medium |
| **Liveness detection** | Prevent spoofing attacks | 🟢 Low (for demo) |

### 7.3 Expected Results After Fixes

With a properly trained CNN:

| Metric | Current | Expected |
|--------|---------|----------|
| Intra-class Hamming | 275 bits | 15-25 bits |
| Inter-class Hamming | ~256 bits | ~255 bits |
| FRR @ 0% noise | 100% | < 1% |
| FRR @ 5% noise | 100% | < 5% |
| FAR | 0% | < 0.01% |
| Effective Security | 0 bits | > 128 bits |

---

## 8. Usage Guide

### 8.1 Installation

```bash
# Full version (requires PyTorch)
pip install -r requirements.txt

# Lightweight version (numpy only)
pip install -r requirements_lite.txt
```

### 8.2 Running Demos

```bash
# Full version
python main.py --mode demo       # Single user demo
python main.py --mode benchmark  # Multi-user benchmark

# Lightweight version
python main_lite.py              # Numpy-only demo
python main_lite.py benchmark    # Numpy-only benchmark
```

### 8.3 Training a Model

```bash
# Download dataset first
# Option 1: CASIA-WebFace (face recognition)
# Option 2: FVC2004 (fingerprint)

python train.py \
    --dataset casia \
    --data_dir ./data/CASIA-WebFace \
    --batch_size 64 \
    --num_epochs 50 \
    --output_dir ./models
```

### 8.4 Evaluation

```bash
python evaluate.py \
    --model ./models/encoder_final.pt \
    --data_dir ./data/test \
    --output ./evaluation
```

---

## Appendix A: Configuration Reference

```python
# config.py defaults
@dataclass
class CNNConfig:
    embedding_dim: int = 512
    arcface_scale: float = 64.0
    arcface_margin: float = 0.5
    input_size: tuple = (112, 112)

@dataclass
class BioHashConfig:
    binary_length: int = 511
    seed: int = 42

@dataclass
class FuzzyExtractorConfig:
    bch_m: int = 9          # GF(2^9) = 512
    bch_t: int = 29         # Correct 29 errors
    key_length: int = 32    # 256-bit output key

@dataclass
class BlockchainConfig:
    provider_url: str = "http://localhost:8545"
    chain_id: int = 1337    # Ganache default
```

---

## Appendix B: References

1. Deng, J., et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019.
2. Dodis, Y., et al. "Fuzzy Extractors: How to Generate Strong Keys from Biometrics." SIAM J. Computing, 2008.
3. Teoh, A.B.J., et al. "Random Multispace Quantisation as an Analytic Mechanism for BioHashing." Pattern Recognition, 2006.
4. Rathgeb, C., Uhl, A. "A Survey on Biometric Cryptosystems and Cancelable Biometrics." EURASIP J. Info Security, 2011.

---

## Appendix C: Glossary

| Term | Definition |
|------|------------|
| **ArcFace** | Angular margin loss that enforces intra-class compactness |
| **BCH Code** | Bose–Chaudhuri–Hocquenghem error-correcting code |
| **BioHashing** | Random projection + binarization for cancelable biometrics |
| **FAR** | False Acceptance Rate (impostor accepted) |
| **FRR** | False Rejection Rate (genuine user rejected) |
| **Fuzzy Extractor** | Cryptographic primitive: Gen(w)→(R,P), Rep(w',P)→R |
| **Helper Data** | Public data P that aids key recovery |
| **Hamming Distance** | Number of bit positions where two strings differ |

---

*Report generated by Claude | Anthropic*