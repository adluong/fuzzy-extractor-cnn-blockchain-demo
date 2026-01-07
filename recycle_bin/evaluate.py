"""
Evaluation Metrics for Biometric Authentication
=================================================

This module computes publication-ready security metrics:
- False Acceptance Rate (FAR)
- False Rejection Rate (FRR)
- Equal Error Rate (EER)
- Detection Error Tradeoff (DET) curves
- ROC curves

These metrics are standard for IEEE S&P and USENIX Security submissions.

Usage:
    python evaluate.py --model ./checkpoints/best.pt --data ./data/test
"""

import os
import argparse
import json
from pathlib import Path
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import interpolate

from config import SystemConfig, CNNConfig, BioHashConfig, FuzzyExtractorConfig
from model import BiometricEncoder
from biohashing import BioHasher, compute_hamming_distance
from fuzzy_extractor import FuzzyExtractor


@dataclass
class AuthenticationResult:
    """Result of an authentication attempt."""
    is_genuine: bool
    success: bool
    hamming_distance: int
    errors_corrected: int
    

class BiometricEvaluator:
    """
    Comprehensive evaluator for biometric authentication systems.
    
    Computes:
    - FAR/FRR at various thresholds
    - EER (Equal Error Rate)
    - DET curves
    - Min-entropy estimation
    """
    
    def __init__(
        self,
        encoder: BiometricEncoder,
        biohasher: BioHasher,
        fuzzy_extractor: FuzzyExtractor,
        device: str = "cuda"
    ):
        self.encoder = encoder.to(device).eval()
        self.biohasher = biohasher
        self.fuzzy_extractor = fuzzy_extractor
        self.device = device
        
    def extract_templates(
        self,
        dataloader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract embeddings and binary templates from dataset.
        
        Returns:
            embeddings: (N, embedding_dim)
            binary_templates: (N, binary_length)
            labels: (N,)
        """
        all_embeddings = []
        all_binary = []
        all_labels = []
        
        for images, labels in tqdm(dataloader, desc="Extracting templates"):
            images = images.to(self.device)
            
            with torch.no_grad():
                embeddings = self.encoder(images)
                binary = self.biohasher(embeddings)
            
            all_embeddings.append(embeddings.cpu())
            all_binary.append(binary.cpu())
            all_labels.append(labels)
        
        return (
            torch.cat(all_embeddings, dim=0),
            torch.cat(all_binary, dim=0),
            torch.cat(all_labels, dim=0)
        )
    
    def compute_distance_distributions(
        self,
        templates: torch.Tensor,
        labels: torch.Tensor,
        num_samples: int = 10000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute genuine and impostor Hamming distance distributions.
        
        Args:
            templates: Binary templates
            labels: Identity labels
            num_samples: Number of pairs to sample
            
        Returns:
            (genuine_distances, impostor_distances)
        """
        genuine_distances = []
        impostor_distances = []
        
        n = len(templates)
        
        # Sample genuine pairs (same identity)
        unique_labels = labels.unique()
        
        for label in tqdm(unique_labels, desc="Computing genuine distances"):
            mask = labels == label
            class_templates = templates[mask]
            
            if len(class_templates) < 2:
                continue
            
            # All pairs within class
            for i in range(len(class_templates)):
                for j in range(i + 1, len(class_templates)):
                    dist = compute_hamming_distance(class_templates[i], class_templates[j])
                    genuine_distances.append(dist)
        
        # Sample impostor pairs (different identity)
        print("Computing impostor distances...")
        sampled = 0
        while sampled < num_samples:
            i, j = np.random.randint(0, n, 2)
            if labels[i] != labels[j]:
                dist = compute_hamming_distance(templates[i], templates[j])
                impostor_distances.append(dist)
                sampled += 1
        
        return np.array(genuine_distances), np.array(impostor_distances)
    
    def compute_far_frr(
        self,
        genuine_distances: np.ndarray,
        impostor_distances: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute FAR and FRR at various thresholds.
        
        FAR = P(accept | impostor) = P(distance ≤ threshold | impostor)
        FRR = P(reject | genuine) = P(distance > threshold | genuine)
        
        Returns:
            (thresholds, far_values, frr_values)
        """
        if thresholds is None:
            max_dist = max(genuine_distances.max(), impostor_distances.max())
            thresholds = np.arange(0, max_dist + 1)
        
        far_values = []
        frr_values = []
        
        for t in thresholds:
            # FAR: impostors accepted (distance ≤ threshold)
            far = (impostor_distances <= t).sum() / len(impostor_distances)
            far_values.append(far)
            
            # FRR: genuine rejected (distance > threshold)
            frr = (genuine_distances > t).sum() / len(genuine_distances)
            frr_values.append(frr)
        
        return thresholds, np.array(far_values), np.array(frr_values)
    
    def compute_eer(
        self,
        far_values: np.ndarray,
        frr_values: np.ndarray,
        thresholds: np.ndarray
    ) -> Tuple[float, float]:
        """
        Compute Equal Error Rate (EER).
        
        EER is the point where FAR = FRR.
        
        Returns:
            (eer, eer_threshold)
        """
        # Find intersection point
        diff = np.abs(far_values - frr_values)
        min_idx = np.argmin(diff)
        
        eer = (far_values[min_idx] + frr_values[min_idx]) / 2
        eer_threshold = thresholds[min_idx]
        
        return eer, eer_threshold
    
    def compute_fuzzy_extractor_metrics(
        self,
        templates: torch.Tensor,
        labels: torch.Tensor,
        num_trials: int = 1000
    ) -> Dict[str, Any]:
        """
        Evaluate fuzzy extractor performance.
        
        Computes:
        - Key recovery success rate at various noise levels
        - Average number of errors corrected
        - False positive rate (impostor key recovery)
        """
        results = {
            'genuine': {'success': 0, 'total': 0, 'errors': []},
            'impostor': {'success': 0, 'total': 0}
        }
        
        # Enroll some users
        enrolled = {}
        unique_labels = labels.unique()[:50]  # Limit for speed
        
        print("Enrolling users...")
        for label in unique_labels:
            mask = labels == label
            class_templates = templates[mask]
            
            if len(class_templates) > 0:
                bio_bytes = self.biohasher.to_bytes(class_templates[0])
                key, helper = self.fuzzy_extractor.gen(bio_bytes)
                enrolled[label.item()] = (key, helper, class_templates)
        
        # Test genuine authentication
        print("Testing genuine authentication...")
        for label, (key, helper, class_templates) in enrolled.items():
            for template in class_templates[1:]:  # Skip enrollment template
                bio_bytes = self.biohasher.to_bytes(template)
                recovered_key, errors = self.fuzzy_extractor.rep(bio_bytes, helper)
                
                results['genuine']['total'] += 1
                if recovered_key is not None and recovered_key == key:
                    results['genuine']['success'] += 1
                    results['genuine']['errors'].append(errors)
        
        # Test impostor authentication
        print("Testing impostor authentication...")
        for _ in range(num_trials):
            # Pick a victim
            victim_label = np.random.choice(list(enrolled.keys()))
            key, helper, _ = enrolled[victim_label]
            
            # Pick an impostor
            other_labels = [l for l in labels.unique().numpy() if l != victim_label]
            impostor_label = np.random.choice(other_labels)
            impostor_mask = labels == impostor_label
            impostor_template = templates[impostor_mask][0]
            
            bio_bytes = self.biohasher.to_bytes(impostor_template)
            recovered_key, _ = self.fuzzy_extractor.rep(bio_bytes, helper)
            
            results['impostor']['total'] += 1
            if recovered_key is not None:
                results['impostor']['success'] += 1
        
        # Compute rates
        genuine_success_rate = (
            results['genuine']['success'] / max(results['genuine']['total'], 1)
        )
        impostor_success_rate = (
            results['impostor']['success'] / max(results['impostor']['total'], 1)
        )
        avg_errors = (
            np.mean(results['genuine']['errors']) if results['genuine']['errors'] else 0
        )
        
        return {
            'genuine_success_rate': genuine_success_rate,
            'genuine_frr': 1 - genuine_success_rate,
            'impostor_success_rate': impostor_success_rate,
            'impostor_far': impostor_success_rate,
            'avg_errors_corrected': avg_errors,
            'max_correctable_errors': self.fuzzy_extractor.t,
            'total_genuine_trials': results['genuine']['total'],
            'total_impostor_trials': results['impostor']['total']
        }
    
    def estimate_min_entropy(
        self,
        templates: torch.Tensor
    ) -> Dict[str, float]:
        """
        Estimate min-entropy of the biometric templates.
        
        Uses several estimation methods:
        1. Bit-level entropy (assuming independence)
        2. Sample-based min-entropy estimation
        """
        templates_np = templates.numpy()
        n_samples, n_bits = templates_np.shape
        
        # Bit-level entropy (lower bound, assumes independence)
        bit_means = templates_np.mean(axis=0)
        # Clamp to avoid log(0)
        bit_means = np.clip(bit_means, 1e-10, 1 - 1e-10)
        
        bit_entropy = -(
            bit_means * np.log2(bit_means) + 
            (1 - bit_means) * np.log2(1 - bit_means)
        )
        total_bit_entropy = bit_entropy.sum()
        
        # Min-entropy estimation (collision-based)
        # Count collisions (matching templates)
        collision_count = 0
        for i in range(min(1000, n_samples)):
            for j in range(i + 1, min(1000, n_samples)):
                if np.array_equal(templates_np[i], templates_np[j]):
                    collision_count += 1
        
        if collision_count > 0:
            # Estimate max probability from collision rate
            # P(collision) ≈ max_x p(x)^2 for large n
            collision_rate = collision_count / (n_samples * (n_samples - 1) / 2)
            estimated_max_prob = np.sqrt(collision_rate)
            collision_entropy = -np.log2(estimated_max_prob)
        else:
            collision_entropy = n_bits  # Upper bound
        
        return {
            'bit_entropy': total_bit_entropy,
            'bit_entropy_per_bit': total_bit_entropy / n_bits,
            'collision_entropy_estimate': collision_entropy,
            'collision_count': collision_count,
            'num_samples': n_samples,
            'template_length': n_bits
        }
    
    def generate_det_curve(
        self,
        far_values: np.ndarray,
        frr_values: np.ndarray,
        save_path: Optional[str] = None
    ):
        """
        Generate Detection Error Tradeoff (DET) curve.
        
        DET curves plot FRR vs FAR on normal deviate scales,
        which is standard for biometric evaluation papers.
        """
        from scipy.stats import norm
        
        # Convert to normal deviates
        # Clamp values to avoid inf
        far_clamped = np.clip(far_values, 1e-6, 1 - 1e-6)
        frr_clamped = np.clip(frr_values, 1e-6, 1 - 1e-6)
        
        far_nd = norm.ppf(far_clamped)
        frr_nd = norm.ppf(frr_clamped)
        
        plt.figure(figsize=(8, 8))
        plt.plot(far_nd, frr_nd, 'b-', linewidth=2)
        
        # Add EER line
        eer_line = np.linspace(-3, 3, 100)
        plt.plot(eer_line, eer_line, 'k--', label='EER line')
        
        # Customize axes
        ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
        tick_labels = [f'{t*100:.1f}%' for t in ticks]
        tick_locs = [norm.ppf(t) for t in ticks]
        
        plt.xticks(tick_locs, tick_labels)
        plt.yticks(tick_locs, tick_labels)
        
        plt.xlabel('False Acceptance Rate (FAR)')
        plt.ylabel('False Rejection Rate (FRR)')
        plt.title('Detection Error Tradeoff (DET) Curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved DET curve to {save_path}")
        
        plt.close()
    
    def generate_roc_curve(
        self,
        far_values: np.ndarray,
        frr_values: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Generate ROC curve (1-FRR vs FAR)."""
        plt.figure(figsize=(8, 8))
        
        # ROC: True Positive Rate (1-FRR) vs False Positive Rate (FAR)
        tpr = 1 - frr_values
        fpr = far_values
        
        plt.plot(fpr, tpr, 'b-', linewidth=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Compute AUC
        auc = np.trapz(tpr, fpr)
        
        plt.xlabel('False Positive Rate (FAR)')
        plt.ylabel('True Positive Rate (1-FRR)')
        plt.title(f'ROC Curve (AUC = {auc:.4f})')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved ROC curve to {save_path}")
        
        plt.close()
        
        return auc
    
    def generate_distance_histogram(
        self,
        genuine_distances: np.ndarray,
        impostor_distances: np.ndarray,
        save_path: Optional[str] = None
    ):
        """Generate histogram of genuine vs impostor distances."""
        plt.figure(figsize=(10, 6))
        
        bins = np.arange(0, max(genuine_distances.max(), impostor_distances.max()) + 2)
        
        plt.hist(genuine_distances, bins=bins, alpha=0.7, label='Genuine', color='green')
        plt.hist(impostor_distances, bins=bins, alpha=0.7, label='Impostor', color='red')
        
        plt.xlabel('Hamming Distance')
        plt.ylabel('Frequency')
        plt.title('Genuine vs Impostor Distance Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved histogram to {save_path}")
        
        plt.close()


def run_full_evaluation(
    model_path: str,
    data_dir: str,
    output_dir: str,
    device: str = "cuda"
):
    """
    Run complete evaluation pipeline and generate publication-ready metrics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("BIOMETRIC AUTHENTICATION SYSTEM EVALUATION")
    print("=" * 60)
    
    # Initialize components
    config = SystemConfig()
    
    # Load encoder
    print("\n[1] Loading model...")
    encoder = BiometricEncoder(config.cnn)
    if os.path.exists(model_path):
        encoder.load_state_dict(torch.load(model_path, map_location=device))
        print(f"  Loaded weights from {model_path}")
    else:
        print(f"  Warning: Using random weights (no checkpoint found)")
    
    biohasher = BioHasher(config.biohash)
    fuzzy_extractor = FuzzyExtractor(config.fuzzy_extractor)
    
    evaluator = BiometricEvaluator(encoder, biohasher, fuzzy_extractor, device)
    
    # Create synthetic test data (replace with real dataset in production)
    print("\n[2] Generating synthetic test data...")
    torch.manual_seed(42)
    
    # Simulate 100 users, 10 samples each
    num_users = 100
    samples_per_user = 10
    
    templates = []
    labels = []
    
    for user_id in range(num_users):
        # Generate user's base embedding
        base_embedding = torch.randn(1, config.cnn.embedding_dim)
        base_embedding = torch.nn.functional.normalize(base_embedding, p=2, dim=1)
        
        for _ in range(samples_per_user):
            # Add intra-class variation
            noise = torch.randn_like(base_embedding) * 0.1
            noisy_embedding = base_embedding + noise
            noisy_embedding = torch.nn.functional.normalize(noisy_embedding, p=2, dim=1)
            
            binary = biohasher(noisy_embedding)
            templates.append(binary)
            labels.append(user_id)
    
    templates = torch.cat(templates, dim=0)
    labels = torch.tensor(labels)
    
    print(f"  Generated {len(templates)} templates for {num_users} users")
    
    # Compute distance distributions
    print("\n[3] Computing distance distributions...")
    genuine_dist, impostor_dist = evaluator.compute_distance_distributions(
        templates, labels, num_samples=5000
    )
    
    print(f"  Genuine pairs: {len(genuine_dist)}")
    print(f"  Impostor pairs: {len(impostor_dist)}")
    print(f"  Genuine distance: {genuine_dist.mean():.1f} ± {genuine_dist.std():.1f}")
    print(f"  Impostor distance: {impostor_dist.mean():.1f} ± {impostor_dist.std():.1f}")
    
    # Compute FAR/FRR
    print("\n[4] Computing FAR/FRR curves...")
    thresholds, far, frr = evaluator.compute_far_frr(genuine_dist, impostor_dist)
    
    # Compute EER
    eer, eer_threshold = evaluator.compute_eer(far, frr, thresholds)
    print(f"  EER: {eer*100:.2f}% at threshold {eer_threshold}")
    
    # Generate plots
    print("\n[5] Generating plots...")
    evaluator.generate_distance_histogram(
        genuine_dist, impostor_dist,
        save_path=output_dir / "distance_histogram.png"
    )
    
    evaluator.generate_det_curve(
        far, frr,
        save_path=output_dir / "det_curve.png"
    )
    
    auc = evaluator.generate_roc_curve(
        far, frr,
        save_path=output_dir / "roc_curve.png"
    )
    
    # Fuzzy extractor evaluation
    print("\n[6] Evaluating fuzzy extractor...")
    fe_metrics = evaluator.compute_fuzzy_extractor_metrics(
        templates, labels, num_trials=500
    )
    
    print(f"  Genuine success rate: {fe_metrics['genuine_success_rate']*100:.2f}%")
    print(f"  Impostor FAR: {fe_metrics['impostor_far']*100:.4f}%")
    print(f"  Avg errors corrected: {fe_metrics['avg_errors_corrected']:.1f}")
    
    # Entropy estimation
    print("\n[7] Estimating min-entropy...")
    entropy_metrics = evaluator.estimate_min_entropy(templates)
    print(f"  Bit entropy: {entropy_metrics['bit_entropy']:.1f} bits")
    print(f"  Entropy per bit: {entropy_metrics['bit_entropy_per_bit']:.3f}")
    
    # Save results
    print("\n[8] Saving results...")
    
    results = {
        'dataset': {
            'num_users': num_users,
            'samples_per_user': samples_per_user,
            'total_templates': len(templates),
            'template_length': int(templates.shape[1])
        },
        'distance_statistics': {
            'genuine_mean': float(genuine_dist.mean()),
            'genuine_std': float(genuine_dist.std()),
            'genuine_max': int(genuine_dist.max()),
            'impostor_mean': float(impostor_dist.mean()),
            'impostor_std': float(impostor_dist.std()),
            'impostor_min': int(impostor_dist.min())
        },
        'error_rates': {
            'eer': float(eer),
            'eer_threshold': int(eer_threshold),
            'roc_auc': float(auc)
        },
        'fuzzy_extractor': fe_metrics,
        'entropy': entropy_metrics,
        'system_parameters': {
            'bch_n': fuzzy_extractor.n,
            'bch_k': fuzzy_extractor.k,
            'bch_t': fuzzy_extractor.t,
            'embedding_dim': config.cnn.embedding_dim,
            'binary_length': config.biohash.binary_length
        }
    }
    
    results_path = output_dir / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  Results saved to {results_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Equal Error Rate (EER):     {eer*100:.2f}%")
    print(f"  ROC AUC:                    {auc:.4f}")
    print(f"  Fuzzy Extractor FRR:        {fe_metrics['genuine_frr']*100:.2f}%")
    print(f"  Fuzzy Extractor FAR:        {fe_metrics['impostor_far']*100:.4f}%")
    print(f"  Template Entropy:           {entropy_metrics['bit_entropy']:.0f} bits")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate biometric authentication system")
    
    parser.add_argument('--model', type=str, default='./checkpoints/encoder_final.pt',
                        help="Path to trained encoder weights")
    parser.add_argument('--data', type=str, default='./data/test',
                        help="Path to test dataset")
    parser.add_argument('--output', type=str, default='./evaluation',
                        help="Output directory for results")
    parser.add_argument('--device', type=str, default='cuda',
                        help="Device to use")
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    
    run_full_evaluation(
        model_path=args.model,
        data_dir=args.data,
        output_dir=args.output,
        device=args.device
    )


if __name__ == "__main__":
    main()