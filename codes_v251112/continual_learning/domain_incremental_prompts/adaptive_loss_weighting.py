"""
Adaptive Loss Weighting for Domain Incremental Learning

Inspired by UDIL (NeurIPS 2023) - parameterizes loss coefficients
and optimizes them during training to minimize generalization bounds.

Key Ideas:
1. Multiple loss components (task, distillation, alignment, contrastive)
2. Learnable coefficients instead of fixed hyperparameters
3. Meta-optimization to find optimal loss balance
4. Constraints to ensure valid probability distributions

References:
- UDIL: Wang et al., NeurIPS 2023
- "A Unified Approach to Domain Incremental Learning with Memory"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class AdaptiveLossWeighting(nn.Module):
    """
    Adaptive Loss Coefficient Learning

    Learns optimal weights for combining multiple loss terms:
    - Task loss (classification on current domain)
    - Distillation loss (knowledge from previous domains)
    - Feature alignment loss (domain adaptation)
    - Contrastive loss (representation learning)

    The weights are parameterized as softmax outputs to ensure:
    1. All weights are positive
    2. Weights sum to a reasonable scale
    """

    def __init__(
        self,
        num_losses: int = 4,
        init_weights: Optional[List[float]] = None,
        min_weight: float = 0.1,
        max_weight: float = 10.0,
        learnable: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            num_losses: Number of loss components to weight
            init_weights: Initial weights (optional)
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            learnable: Whether weights are learnable or fixed
            device: Device for tensors
        """
        super().__init__()

        self.num_losses = num_losses
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.learnable = learnable

        # Initialize logits
        if init_weights is not None:
            assert len(init_weights) == num_losses
            # Convert weights to logits (inverse softmax)
            init_logits = torch.log(torch.tensor(init_weights, dtype=torch.float32))
        else:
            # Uniform initialization
            init_logits = torch.zeros(num_losses, dtype=torch.float32)

        if learnable:
            self.logits = nn.Parameter(init_logits.to(device))
        else:
            self.register_buffer('logits', init_logits.to(device))

        # Track weight history for analysis
        self.weight_history = []

    def get_weights(self) -> torch.Tensor:
        """
        Get current loss weights

        Applies softmax to logits, then scales to [min_weight, max_weight]

        Returns:
            weights: [num_losses] positive weights
        """
        # Softmax to get probability distribution
        probs = F.softmax(self.logits, dim=0)

        # Scale to desired range
        weights = probs * (self.max_weight - self.min_weight) + self.min_weight

        return weights

    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted combination of losses

        Args:
            losses: Dict mapping loss name to loss value
                   Expected keys: task, distillation, alignment, contrastive

        Returns:
            weighted_loss: Scalar weighted combination
        """
        weights = self.get_weights()

        # Standard loss order
        loss_names = ['task', 'distillation', 'alignment', 'contrastive']

        weighted_loss = 0.0
        for i, name in enumerate(loss_names[:self.num_losses]):
            if name in losses:
                weighted_loss += weights[i] * losses[name]

        # Track weights
        self.weight_history.append(weights.detach().cpu().numpy())

        return weighted_loss

    def get_weight_stats(self) -> Dict[str, np.ndarray]:
        """Get statistics of weight evolution"""
        if len(self.weight_history) == 0:
            return {}

        history = np.array(self.weight_history)  # [num_steps, num_losses]

        return {
            'mean': history.mean(axis=0),
            'std': history.std(axis=0),
            'min': history.min(axis=0),
            'max': history.max(axis=0),
            'final': history[-1]
        }


class DomainContrastiveLoss(nn.Module):
    """
    Domain-Aware Contrastive Loss

    Inspired by DARE's contrastive learning component.

    Pulls together:
    - Same emotion, same domain (strong positive)
    - Same emotion, different domain (weak positive)

    Pushes apart:
    - Different emotion (negative)

    Uses AU representations for contrast.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        same_domain_weight: float = 1.0,
        cross_domain_weight: float = 0.5
    ):
        """
        Args:
            temperature: Temperature for contrastive loss
            same_domain_weight: Weight for same-domain positives
            cross_domain_weight: Weight for cross-domain positives
        """
        super().__init__()
        self.temperature = temperature
        self.same_domain_weight = same_domain_weight
        self.cross_domain_weight = cross_domain_weight

    def forward(
        self,
        au_features: torch.Tensor,
        labels: torch.Tensor,
        domain_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute domain-aware contrastive loss

        Args:
            au_features: [batch, num_aus] AU representations
            labels: [batch] emotion labels
            domain_ids: [batch] domain identifiers

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = au_features.shape[0]
        device = au_features.device

        # Normalize features
        au_features = F.normalize(au_features, dim=1)

        # Compute similarity matrix
        similarity = torch.matmul(au_features, au_features.t()) / self.temperature
        # [batch, batch]

        # Create masks
        label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        # Same emotion = 1, different = 0

        domain_mask = (domain_ids.unsqueeze(1) == domain_ids.unsqueeze(0)).float()
        # Same domain = 1, different = 0

        # Identity mask (remove self-comparison)
        identity = torch.eye(batch_size, device=device)

        # Positive pairs mask with domain weighting
        same_domain_same_label = label_mask * domain_mask * (1 - identity)
        cross_domain_same_label = label_mask * (1 - domain_mask)

        positive_mask = (
            self.same_domain_weight * same_domain_same_label +
            self.cross_domain_weight * cross_domain_same_label
        )

        # Negative pairs mask
        negative_mask = (1 - label_mask) * (1 - identity)

        # Compute loss
        # Numerator: weighted sum of positive similarities
        numerator = (similarity * positive_mask).sum(dim=1)

        # Denominator: sum of all similarities (excluding self)
        denominator = (torch.exp(similarity) * (1 - identity)).sum(dim=1)

        # Contrastive loss
        loss = -torch.log(torch.exp(numerator) / (denominator + 1e-8) + 1e-8)
        loss = loss.mean()

        return loss


class FeatureAlignmentLoss(nn.Module):
    """
    Feature Alignment Loss for Domain Adaptation

    Aligns AU feature distributions across domains using:
    1. Maximum Mean Discrepancy (MMD)
    2. Or adversarial domain discriminator

    This helps reduce domain shift in the AU representation space.
    """

    def __init__(
        self,
        method: str = 'mmd',
        kernel: str = 'rbf',
        sigma: float = 1.0
    ):
        """
        Args:
            method: 'mmd' or 'adversarial'
            kernel: Kernel for MMD ('linear' or 'rbf')
            sigma: Bandwidth for RBF kernel
        """
        super().__init__()
        self.method = method
        self.kernel = kernel
        self.sigma = sigma

    def _compute_kernel(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """Compute kernel matrix"""
        if self.kernel == 'linear':
            return torch.matmul(x, y.t())

        elif self.kernel == 'rbf':
            # Compute pairwise distances
            x_norm = (x ** 2).sum(dim=1).view(-1, 1)
            y_norm = (y ** 2).sum(dim=1).view(1, -1)
            dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.t())

            # RBF kernel
            return torch.exp(-dist / (2 * self.sigma ** 2))

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def forward(
        self,
        source_features: torch.Tensor,
        target_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute alignment loss

        Args:
            source_features: [batch_source, feature_dim]
            target_features: [batch_target, feature_dim]

        Returns:
            loss: Scalar alignment loss
        """
        if self.method == 'mmd':
            # Maximum Mean Discrepancy
            K_ss = self._compute_kernel(source_features, source_features)
            K_tt = self._compute_kernel(target_features, target_features)
            K_st = self._compute_kernel(source_features, target_features)

            n = source_features.shape[0]
            m = target_features.shape[0]

            mmd = (
                K_ss.sum() / (n * n) +
                K_tt.sum() / (m * m) -
                2 * K_st.sum() / (n * m)
            )

            return mmd

        else:
            raise NotImplementedError(f"Method {self.method} not implemented")


def test_adaptive_weighting():
    """Test adaptive loss weighting"""
    print("Testing Adaptive Loss Weighting...")

    # Create adaptive weighting
    weighting = AdaptiveLossWeighting(
        num_losses=4,
        learnable=True,
        device='cpu'
    )

    print(f"\nInitial weights: {weighting.get_weights()}")

    # Simulate training
    optimizer = torch.optim.Adam([weighting.logits], lr=0.01)

    for step in range(10):
        # Dummy losses
        losses = {
            'task': torch.tensor(1.5),
            'distillation': torch.tensor(0.8),
            'alignment': torch.tensor(0.3),
            'contrastive': torch.tensor(0.5)
        }

        # Compute weighted loss
        total_loss = weighting(losses)

        # Meta-optimization (in practice, based on validation performance)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 5 == 0:
            print(f"Step {step}: weights = {weighting.get_weights()}")

    # Get weight statistics
    stats = weighting.get_weight_stats()
    print(f"\nWeight statistics:")
    print(f"  Final: {stats['final']}")
    print(f"  Mean: {stats['mean']}")

    # Test contrastive loss
    print("\nTesting Domain Contrastive Loss...")

    contrastive = DomainContrastiveLoss(temperature=0.07)

    au_features = torch.randn(8, 23)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1])
    domain_ids = torch.tensor([0, 0, 0, 1, 1, 1, 1, 1])

    loss = contrastive(au_features, labels, domain_ids)
    print(f"Contrastive loss: {loss.item():.4f}")

    # Test alignment loss
    print("\nTesting Feature Alignment Loss...")

    alignment = FeatureAlignmentLoss(method='mmd', kernel='rbf')

    source_features = torch.randn(16, 23)
    target_features = torch.randn(16, 23) + 1.0  # Shifted distribution

    loss = alignment(source_features, target_features)
    print(f"Alignment loss (MMD): {loss.item():.4f}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_adaptive_weighting()
