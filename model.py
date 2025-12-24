"""
CNN Feature Extractor with ArcFace Loss
========================================

This module implements a ResNet-50 backbone with ArcFace loss for learning
discriminative biometric embeddings suitable for cryptographic key generation.

Key Design Decisions:
- ArcFace margin ensures angular separation > Hamming distance tolerance
- Embedding normalization enables stable binarization
- Gradient scaling prevents training instability
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from typing import Tuple, Optional

from config import CNNConfig, DEFAULT_CONFIG


class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss (ArcFace) for discriminative embedding learning.
    
    Reference: Deng et al., "ArcFace: Additive Angular Margin Loss for Deep 
               Face Recognition", CVPR 2019.
    
    The loss adds an angular margin m to the target angle θ:
        L = -log(exp(s·cos(θ_y + m)) / (exp(s·cos(θ_y + m)) + Σ_{j≠y} exp(s·cos(θ_j))))
    
    This enforces a geodesic distance margin on the hypersphere, which translates
    to better Hamming distance separation after binarization.
    
    Attributes:
        weight: Learnable class center embeddings, shape (num_classes, embedding_dim)
        s: Scale factor (typically 64)
        m: Angular margin in radians (typically 0.5 ≈ 28.6°)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = scale
        self.m = margin
        self.easy_margin = easy_margin
        
        # Class center embeddings (learned)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute trigonometric constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)  # Threshold for numerical stability
        self.mm = math.sin(math.pi - margin) * margin
        
    def forward(
        self, 
        embeddings: torch.Tensor, 
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.
        
        Args:
            embeddings: L2-normalized feature vectors, shape (batch, embedding_dim)
            labels: Ground truth class indices, shape (batch,)
            
        Returns:
            Cross-entropy loss with angular margin applied to target logits
        """
        # Normalize weights
        normalized_weights = F.normalize(self.weight, p=2, dim=1)
        
        # Compute cosine similarity: cos(θ) = <x, w> for normalized vectors
        cosine = F.linear(embeddings, normalized_weights)
        
        # Compute sin(θ) from cos(θ) using identity: sin²θ + cos²θ = 1
        sine = torch.sqrt(1.0 - torch.clamp(cosine ** 2, 0, 1))
        
        # Compute cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m
        
        # Handle edge case when θ + m > π (numerical stability)
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        # Create one-hot encoding for target classes
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        
        # Apply margin only to the target class
        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits *= self.s
        
        return F.cross_entropy(logits, labels)


class BiometricEncoder(nn.Module):
    """
    ResNet-50 based biometric feature extractor.
    
    Architecture:
        Input (112x112x3) → ResNet-50 (modified) → FC → L2-Norm → Embedding (512-D)
    
    The final embedding is L2-normalized to lie on the unit hypersphere,
    which is required for ArcFace loss and improves binarization stability.
    """
    
    def __init__(self, config: CNNConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG.cnn
        
        # Load pretrained ResNet-50 backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Remove the final FC layer (we add our own)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        
        # Modify first conv for 112x112 input (optional, works with 224x224 too)
        # self.features[0] = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        # Bottleneck dimension from ResNet-50
        backbone_dim = 2048
        
        # Embedding head: FC + BatchNorm
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, self.config.embedding_dim),
            nn.BatchNorm1d(self.config.embedding_dim),
        )
        
        # Initialize embedding layer
        nn.init.kaiming_normal_(self.embedding_head[1].weight)
        
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Extract biometric embedding from input image.
        
        Args:
            x: Input images, shape (batch, 3, H, W)
            normalize: If True, L2-normalize the output embedding
            
        Returns:
            Embedding vectors, shape (batch, embedding_dim)
        """
        # Extract backbone features
        features = self.features(x)
        
        # Map to embedding space
        embedding = self.embedding_head(features)
        
        # L2 normalize for hypersphere projection
        if normalize:
            embedding = F.normalize(embedding, p=2, dim=1)
            
        return embedding


class BiometricModel(nn.Module):
    """
    Complete biometric recognition model combining encoder and ArcFace loss.
    
    This is the training wrapper that computes both embeddings and loss.
    For inference (key generation), use only the encoder.
    """
    
    def __init__(self, config: CNNConfig = None):
        super().__init__()
        self.config = config or DEFAULT_CONFIG.cnn
        
        # Feature extractor
        self.encoder = BiometricEncoder(self.config)
        
        # ArcFace classification head (only for training)
        self.arcface = ArcFaceLoss(
            embedding_dim=self.config.embedding_dim,
            num_classes=self.config.num_classes,
            scale=self.config.arcface_scale,
            margin=self.config.arcface_margin
        )
        
    def forward(
        self, 
        x: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training or inference.
        
        Args:
            x: Input images, shape (batch, 3, H, W)
            labels: Ground truth labels for training (optional)
            
        Returns:
            embeddings: Feature vectors, shape (batch, embedding_dim)
            loss: ArcFace loss if labels provided, else None
        """
        embeddings = self.encoder(x)
        
        loss = None
        if labels is not None:
            loss = self.arcface(embeddings, labels)
            
        return embeddings, loss
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embedding for inference (no gradient)."""
        self.eval()
        with torch.no_grad():
            return self.encoder(x)


def create_model(config: CNNConfig = None, pretrained_path: Optional[str] = None) -> BiometricModel:
    """
    Factory function to create and optionally load a pretrained model.
    
    Args:
        config: Model configuration
        pretrained_path: Path to saved model weights
        
    Returns:
        Initialized BiometricModel
    """
    model = BiometricModel(config)
    
    if pretrained_path:
        state_dict = torch.load(pretrained_path, map_location='cpu')
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    return model


# ============================================================================
# Training Utilities
# ============================================================================

def train_epoch(
    model: BiometricModel,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str = "cuda"
) -> float:
    """
    Train for one epoch.
    
    Returns:
        Average loss over the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        _, loss = model(images, labels)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(
    model: BiometricModel,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cuda"
) -> Tuple[float, float]:
    """
    Evaluate model on validation set.
    
    Returns:
        (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        
        embeddings, loss = model(images, labels)
        total_loss += loss.item()
        
        # Compute classification accuracy using ArcFace weights
        normalized_weights = F.normalize(model.arcface.weight, p=2, dim=1)
        logits = F.linear(embeddings, normalized_weights)
        predictions = logits.argmax(dim=1)
        
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy


if __name__ == "__main__":
    # Quick test
    config = CNNConfig(num_classes=100)  # Small test
    model = create_model(config)
    
    # Dummy input
    x = torch.randn(4, 3, 112, 112)
    labels = torch.tensor([0, 1, 2, 3])
    
    embeddings, loss = model(x, labels)
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Embedding norm: {embeddings.norm(dim=1)}")  # Should be ~1.0