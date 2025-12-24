"""
Configuration for CNN + Fuzzy Extractor + Blockchain System
============================================================

This module centralizes all hyperparameters and system configurations.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class CNNConfig:
    """Configuration for the CNN feature extractor."""
    # Model architecture
    backbone: str = "resnet50"
    embedding_dim: int = 512
    num_classes: int = 10575  # CASIA-WebFace has ~10k identities
    
    # ArcFace hyperparameters
    arcface_scale: float = 64.0  # s: scaling factor
    arcface_margin: float = 0.5  # m: angular margin in radians (~28.6 degrees)
    
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 5e-4
    num_epochs: int = 50
    
    # Input dimensions
    input_height: int = 112
    input_width: int = 112


@dataclass
class BioHashConfig:
    """Configuration for BioHashing layer."""
    # Binary code length (must be compatible with BCH code)
    # BCH(511, 277) can correct up to 29 errors
    binary_length: int = 511
    
    # Random projection seed (user-specific in production)
    projection_seed: Optional[int] = None
    
    # Embedding dimension (must match CNN output)
    embedding_dim: int = 512


@dataclass
class FuzzyExtractorConfig:
    """Configuration for BCH-based Fuzzy Extractor."""
    # BCH code parameters
    # n = 2^m - 1 (codeword length)
    # t = error correction capability
    bch_polynomial: int = 0b10000001001  # Primitive polynomial for m=9
    bch_m: int = 9        # GF(2^m), n = 2^9 - 1 = 511
    bch_t: int = 29       # Can correct up to 29 bit errors (~5.7% error rate)
    
    # Key length in bits (derived from BCH parameters)
    # k = n - m*t for BCH
    key_length_bits: int = 250  # Approximate, actual depends on BCH construction
    
    # Security parameter
    min_entropy_threshold: float = 128.0  # Minimum acceptable entropy bits


@dataclass
class BlockchainConfig:
    """Configuration for Ethereum blockchain integration."""
    # Network settings
    provider_url: str = "http://127.0.0.1:8545"  # Local Ganache/Hardhat
    chain_id: int = 1337  # Local development chain
    
    # Contract settings
    gas_limit: int = 3000000
    gas_price_gwei: int = 20
    
    # Contract address (set after deployment)
    contract_address: Optional[str] = None


@dataclass
class SystemConfig:
    """Master configuration combining all components."""
    cnn: CNNConfig = None
    biohash: BioHashConfig = None
    fuzzy_extractor: FuzzyExtractorConfig = None
    blockchain: BlockchainConfig = None
    
    # Paths
    data_dir: str = "./data"
    model_dir: str = "./models"
    log_dir: str = "./logs"
    
    # Device
    device: str = "cuda"  # or "cpu"
    
    def __post_init__(self):
        self.cnn = self.cnn or CNNConfig()
        self.biohash = self.biohash or BioHashConfig()
        self.fuzzy_extractor = self.fuzzy_extractor or FuzzyExtractorConfig()
        self.blockchain = self.blockchain or BlockchainConfig()
        
        # Ensure directories exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = SystemConfig()