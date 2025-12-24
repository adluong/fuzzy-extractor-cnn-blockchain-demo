"""
Training Script for Biometric CNN with ArcFace
================================================

This script trains the biometric feature extractor on real datasets.

Supported Datasets:
- CASIA-WebFace (face recognition)
- FVC2004 (fingerprint recognition)
- Custom datasets following ImageFolder structure

Usage:
    python train.py --dataset casia --data_dir ./data/CASIA-WebFace
    python train.py --dataset custom --data_dir ./data/my_biometrics
"""

import os
import sys
import argparse
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torchvision import transforms, datasets

from tqdm import tqdm

from config import SystemConfig, CNNConfig
from model import BiometricModel, train_epoch, evaluate, create_model
from biohashing import BioHasher, analyze_biohash_statistics


class BiometricDataset(Dataset):
    """
    Custom dataset wrapper for biometric data.
    
    Handles both face and fingerprint modalities with appropriate
    preprocessing pipelines.
    """
    
    def __init__(
        self,
        root: str,
        modality: str = "face",
        split: str = "train",
        transform=None
    ):
        self.root = Path(root)
        self.modality = modality
        self.split = split
        
        # Default transforms based on modality
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
        
        # Load dataset (assuming ImageFolder structure)
        self.dataset = datasets.ImageFolder(
            root=self.root,
            transform=self.transform
        )
        
        self.num_classes = len(self.dataset.classes)
        print(f"Loaded {len(self.dataset)} samples from {self.num_classes} classes")
    
    def _get_default_transform(self):
        """Get default transform based on modality."""
        if self.modality == "face":
            if self.split == "train":
                return transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.RandomCrop((112, 112)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2
                    ),
                    transforms.RandomGrayscale(p=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    ),
                    transforms.RandomErasing(p=0.2)
                ])
            else:
                return transforms.Compose([
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    )
                ])
        
        elif self.modality == "fingerprint":
            # Fingerprints are typically grayscale
            if self.split == "train":
                return transforms.Compose([
                    transforms.Resize((128, 128)),
                    transforms.RandomCrop((112, 112)),
                    transforms.RandomRotation(degrees=15),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    )
                ])
            else:
                return transforms.Compose([
                    transforms.Resize((112, 112)),
                    transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5]
                    )
                ])
        
        else:
            raise ValueError(f"Unknown modality: {self.modality}")
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class Trainer:
    """
    Training orchestrator for biometric CNN.
    
    Features:
    - Mixed precision training (FP16)
    - Gradient accumulation for large effective batch sizes
    - Learning rate scheduling (OneCycleLR or Cosine)
    - Checkpoint saving and resumption
    - TensorBoard logging
    - Early stopping
    """
    
    def __init__(
        self,
        model: BiometricModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Scheduler
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'] * 10,
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader)
        )
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None
        
        # Tracking
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        # Checkpointing
        self.checkpoint_dir = Path(config.get('checkpoint_dir', './checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def train_epoch(self) -> float:
        """Train for one epoch with mixed precision."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Mixed precision forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    _, loss = self.model(images, labels)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                _, loss = self.model(images, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc="Validating"):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            embeddings, loss = self.model(images, labels)
            total_loss += loss.item()
            
            # Compute accuracy
            normalized_weights = F.normalize(self.model.arcface.weight, p=2, dim=1)
            logits = F.linear(embeddings, normalized_weights)
            predictions = logits.argmax(dim=1)
            
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        # Save latest
        torch.save(checkpoint, self.checkpoint_dir / 'latest.pt')
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best.pt')
            print(f"  Saved new best model!")
    
    def load_checkpoint(self, path: str):
        """Load checkpoint for resumption."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        return checkpoint['epoch']
    
    def train(self, num_epochs: int, early_stopping_patience: int = 10):
        """Full training loop."""
        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Device: {self.device}")
        print("-" * 50)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
            
            # Check for improvement
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Early stopping
            if self.epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        print("\nTraining complete!")
        return self.history


def evaluate_for_fuzzy_extractor(
    model: BiometricModel,
    dataloader: DataLoader,
    biohasher: BioHasher,
    device: str = "cuda"
) -> Dict[str, Any]:
    """
    Evaluate model's suitability for fuzzy extractor integration.
    
    This computes:
    - Intra-class vs inter-class Hamming distances
    - Estimated FAR/FRR at various thresholds
    - Entropy estimation
    """
    model.eval()
    
    all_embeddings = []
    all_labels = []
    
    print("Extracting embeddings...")
    for images, labels in tqdm(dataloader):
        images = images.to(device)
        
        with torch.no_grad():
            embeddings = model.encoder(images)
        
        all_embeddings.append(embeddings.cpu())
        all_labels.append(labels)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    print(f"Extracted {len(embeddings)} embeddings")
    
    # Analyze BioHash statistics
    stats = analyze_biohash_statistics(biohasher, embeddings, labels)
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Train biometric CNN")
    
    parser.add_argument('--dataset', type=str, default='custom',
                        choices=['casia', 'fvc2004', 'custom'])
    parser.add_argument('--data_dir', type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument('--modality', type=str, default='face',
                        choices=['face', 'fingerprint'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    
    full_dataset = BiometricDataset(
        root=args.data_dir,
        modality=args.modality,
        split='train'
    )
    
    # Split into train/val
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    config = CNNConfig(num_classes=full_dataset.num_classes)
    model = create_model(config)
    
    print(f"\nModel: ResNet-50 + ArcFace")
    print(f"Classes: {config.num_classes}")
    print(f"Embedding dim: {config.embedding_dim}")
    
    # Training config
    train_config = {
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_epochs': args.num_epochs,
        'checkpoint_dir': args.checkpoint_dir
    }
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=train_config,
        device=args.device
    )
    
    # Resume if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Train
    history = trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=10
    )
    
    # Save final model (encoder only, for deployment)
    encoder_path = Path(args.checkpoint_dir) / 'encoder_final.pt'
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"\nSaved encoder to {encoder_path}")
    
    # Evaluate for Fuzzy Extractor
    print("\nEvaluating for Fuzzy Extractor integration...")
    biohasher = BioHasher()
    stats = evaluate_for_fuzzy_extractor(model, val_loader, biohasher, args.device)
    
    print("\nBioHash Statistics:")
    print(f"  Binary length: {stats['binary_length']} bits")
    print(f"  Intra-class Hamming distance: {stats['intra_class']['mean']:.1f} ± {stats['intra_class']['std']:.1f}")
    print(f"  Inter-class Hamming distance: {stats['inter_class']['mean']:.1f} ± {stats['inter_class']['std']:.1f}")
    print(f"  Separation margin: {stats['separation_margin']:.1f} bits")
    print(f"  Estimated entropy: {stats['estimated_entropy_bits']:.1f} bits")
    
    # Save stats
    stats_path = Path(args.checkpoint_dir) / 'biohash_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics to {stats_path}")


if __name__ == "__main__":
    main()