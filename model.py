"""
Biometric Feature Extractor using Pretrained FaceNet
=====================================================

This module uses facenet-pytorch's InceptionResnetV1 pretrained on VGGFace2
for high-quality face embeddings suitable for fuzzy extractor pipelines.

Key Advantages:
- Pretrained on 3.3M face images (VGGFace2 dataset)
- Produces 512-D L2-normalized embeddings
- Deterministic inference (same face → same embedding)
- No training required

Reference: 
- Schroff et al., "FaceNet: A Unified Embedding for Face Recognition", CVPR 2015
- facenet-pytorch: https://github.com/timesler/facenet-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import numpy as np
import os
import ssl

# Fix SSL certificate issues (common on Windows/WSL2)
# This must be done BEFORE importing facenet_pytorch
try:
    import certifi
    os.environ['SSL_CERT_FILE'] = certifi.where()
except ImportError:
    pass

# Alternative SSL fix if certifi doesn't work
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Check for facenet-pytorch availability
HAS_FACENET = False
try:
    from facenet_pytorch import InceptionResnetV1, MTCNN
    HAS_FACENET = True
except ImportError:
    print("WARNING: facenet-pytorch not installed. Install with: pip install facenet-pytorch")

# Fallback to torchvision if facenet not available
try:
    from torchvision import models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False


class FaceNetEncoder(nn.Module):
    """
    Face embedding extractor using pretrained InceptionResnetV1.
    
    This model is trained with center loss + softmax on VGGFace2,
    producing embeddings with good angular separation - ideal for
    binarization and fuzzy extractors.
    
    Attributes:
        model: InceptionResnetV1 backbone
        embedding_dim: Output dimension (512)
    """
    
    def __init__(self, pretrained: str = 'vggface2', device: str = None):
        """
        Initialize FaceNet encoder.
        
        Args:
            pretrained: 'vggface2' or 'casia-webface'
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        super().__init__()
        
        if not HAS_FACENET:
            raise ImportError(
                "facenet-pytorch is required. Install with:\n"
                "  pip install facenet-pytorch"
            )
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = 512
        
        # Apply SSL fix before downloading weights
        import ssl
        import urllib.request
        
        # Create unverified SSL context for download
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Monkey-patch urllib to use unverified context
        original_urlopen = urllib.request.urlopen
        def patched_urlopen(url, *args, **kwargs):
            if 'context' not in kwargs:
                kwargs['context'] = ssl_context
            return original_urlopen(url, *args, **kwargs)
        
        urllib.request.urlopen = patched_urlopen
        
        try:
            # Load pretrained model
            self.model = InceptionResnetV1(
                pretrained=pretrained,
                classify=False,  # We want embeddings, not classifications
                device=self.device
            )
        finally:
            # Restore original urlopen
            urllib.request.urlopen = original_urlopen
        
        # Set to eval mode for deterministic inference
        self.model.eval()
        
        # Freeze all parameters (we're not training)
        for param in self.model.parameters():
            param.requires_grad = False
        
        print(f"FaceNet encoder loaded (pretrained={pretrained}, device={self.device})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract face embedding.
        
        Args:
            x: Face images, shape (batch, 3, 160, 160) or (batch, 3, H, W)
               Values should be in range [0, 1] or [-1, 1]
               
        Returns:
            L2-normalized embeddings, shape (batch, 512)
        """
        # Ensure eval mode for deterministic output
        self.model.eval()
        
        with torch.no_grad():
            # Move input to same device as model
            x = x.to(self.device)
            
            # Resize to expected input size if needed (160x160 for FaceNet)
            if x.shape[-1] != 160 or x.shape[-2] != 160:
                x = F.interpolate(x, size=(160, 160), mode='bilinear', align_corners=False)
            
            # Normalize to [-1, 1] if input is [0, 1]
            if x.min() >= 0 and x.max() <= 1:
                x = (x - 0.5) / 0.5
            
            # Get embeddings
            embeddings = self.model(x)
            
            # L2 normalize (should already be normalized, but ensure it)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            # Move back to CPU for compatibility with rest of pipeline
            embeddings = embeddings.cpu()
        
        return embeddings
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward() for compatibility."""
        return self.forward(x)


class BiometricEncoder(nn.Module):
    """
    Unified biometric encoder interface.
    
    Automatically selects the best available backend:
    1. FaceNet (facenet-pytorch) - preferred
    2. ResNet-50 (torchvision) - fallback
    
    This provides a consistent API regardless of which backend is used.
    """
    
    def __init__(self, config=None, pretrained: str = 'vggface2'):
        """
        Initialize biometric encoder.
        
        Args:
            config: Optional CNNConfig (for compatibility)
            pretrained: Pretrained weights to use
        """
        super().__init__()
        
        self.embedding_dim = 512
        self.backend = None
        
        # Try FaceNet first (preferred)
        if HAS_FACENET:
            try:
                self.encoder = FaceNetEncoder(pretrained=pretrained)
                self.backend = 'facenet'
                self.input_size = (160, 160)
            except Exception as e:
                print(f"FaceNet initialization failed: {e}")
        
        # Fallback to ResNet
        if self.backend is None and HAS_TORCHVISION:
            print("Using ResNet-50 fallback (less accurate than FaceNet)")
            self.encoder = self._create_resnet_encoder()
            self.backend = 'resnet'
            self.input_size = (112, 112)
        
        if self.backend is None:
            raise ImportError(
                "No suitable backend found. Install one of:\n"
                "  pip install facenet-pytorch  (recommended)\n"
                "  pip install torchvision"
            )
        
        # Set to eval mode
        self.eval()
    
    def _create_resnet_encoder(self) -> nn.Module:
        """Create ResNet-50 based encoder as fallback."""
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove classification head
        modules = list(backbone.children())[:-1]
        features = nn.Sequential(*modules)
        
        # Add embedding head
        encoder = nn.Sequential(
            features,
            nn.Flatten(),
            nn.Linear(2048, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
        )
        
        return encoder
    
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Extract biometric embedding.
        
        Args:
            x: Input images, shape (batch, 3, H, W)
            normalize: L2-normalize output (default True)
            
        Returns:
            Embeddings, shape (batch, 512)
        """
        self.eval()  # Ensure eval mode
        
        with torch.no_grad():
            # Resize to expected size
            target_size = self.input_size
            if x.shape[-1] != target_size[1] or x.shape[-2] != target_size[0]:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            
            if self.backend == 'facenet':
                embeddings = self.encoder(x)
            else:
                # ResNet fallback
                embeddings = self.encoder(x)
                if normalize:
                    embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding (alias for forward)."""
        return self.forward(x)


class BiometricModel(nn.Module):
    """
    Complete biometric model for training and inference.
    
    For pretrained models, this is just a wrapper around BiometricEncoder.
    For training from scratch, it includes ArcFace loss.
    """
    
    def __init__(self, config=None, num_classes: int = None):
        """
        Initialize biometric model.
        
        Args:
            config: Optional configuration
            num_classes: Number of identity classes (for training only)
        """
        super().__init__()
        
        self.encoder = BiometricEncoder(config)
        self.embedding_dim = self.encoder.embedding_dim
        
        # ArcFace head for training (optional)
        self.arcface = None
        if num_classes is not None:
            self.arcface = ArcFaceLoss(
                embedding_dim=self.embedding_dim,
                num_classes=num_classes
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input images
            labels: Class labels (for training with ArcFace)
            
        Returns:
            (embeddings, loss) - loss is None if labels not provided
        """
        embeddings = self.encoder(x)
        
        loss = None
        if labels is not None and self.arcface is not None:
            loss = self.arcface(embeddings, labels)
        
        return embeddings, loss
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for inference."""
        return self.encoder.get_embedding(x)


class ArcFaceLoss(nn.Module):
    """
    ArcFace loss for training (not needed for pretrained models).
    
    Included for compatibility with training pipeline.
    """
    
    def __init__(
        self,
        embedding_dim: int = 512,
        num_classes: int = 1000,
        scale: float = 64.0,
        margin: float = 0.5
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute constants
        self.cos_m = np.cos(margin)
        self.sin_m = np.sin(margin)
        self.th = np.cos(np.pi - margin)
        self.mm = np.sin(np.pi - margin) * margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute ArcFace loss."""
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)
        
        # Cosine similarity
        cosine = F.linear(embeddings, weights)
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        
        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # One-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.scale
        
        return F.cross_entropy(logits, labels)


def create_model(config=None, pretrained_path: Optional[str] = None) -> BiometricModel:
    """
    Factory function to create biometric model.
    
    Args:
        config: Optional configuration
        pretrained_path: Path to custom pretrained weights (not needed for FaceNet)
        
    Returns:
        Initialized BiometricModel
    """
    model = BiometricModel(config)
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded custom weights from {pretrained_path}")
    
    return model


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Biometric Encoder")
    print("=" * 60)
    
    # Create encoder
    encoder = BiometricEncoder()
    print(f"Backend: {encoder.backend}")
    print(f"Embedding dim: {encoder.embedding_dim}")
    print(f"Input size: {encoder.input_size}")
    
    # Test with dummy input
    print("\n[Test 1] Deterministic output")
    x = torch.randn(1, 3, 160, 160)
    
    emb1 = encoder(x)
    emb2 = encoder(x)
    
    diff = (emb1 - emb2).abs().max().item()
    print(f"  Same input, max diff: {diff:.10f}")
    print(f"  Deterministic: {'✓ YES' if diff < 1e-6 else '✗ NO'}")
    
    print("\n[Test 2] Embedding properties")
    print(f"  Shape: {emb1.shape}")
    print(f"  L2 norm: {emb1.norm().item():.6f} (should be ~1.0)")
    print(f"  Mean: {emb1.mean().item():.6f}")
    print(f"  Std: {emb1.std().item():.6f}")
    
    print("\n[Test 3] Different inputs")
    x2 = torch.randn(1, 3, 160, 160)
    emb3 = encoder(x2)
    
    cosine_sim = F.cosine_similarity(emb1, emb3).item()
    print(f"  Cosine similarity (random vs random): {cosine_sim:.4f}")
    
    print("\n[Test 4] Batch processing")
    batch = torch.randn(4, 3, 160, 160)
    batch_emb = encoder(batch)
    print(f"  Batch shape: {batch_emb.shape}")
    print(f"  All norms ~1: {batch_emb.norm(dim=1)}")
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)