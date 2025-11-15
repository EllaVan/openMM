"""
AU-EMO Probability Matrix for Continual Learning

This module implements a Bayesian-updatable AU-EMO probability matrix
that maintains associations between Action Units and Emotions across
multiple domains in continual learning scenarios.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import json
from pathlib import Path


class AUEMOMatrix(nn.Module):
    """
    AU-EMO Probability Matrix with Bayesian Update

    This class maintains P(EMO|AU) associations and updates them using
    Bayesian inference. It uses a Dirichlet-Multinomial conjugate prior
    framework for stable probability updates.

    Parameters:
    -----------
    num_aus : int
        Number of Action Units (e.g., 23)
    num_emotions : int
        Number of emotion classes (excluding neutral, e.g., 6)
    prior_matrix : torch.Tensor or np.ndarray, optional
        Psychology prior matrix [num_aus, num_emotions]
        If None, uniform distribution is used
    prior_strength : float
        Strength of the prior (pseudo-count multiplier)
        Higher values make the matrix更resistant to updates
    device : str
        Device to store tensors ('cuda' or 'cpu')

    Attributes:
    -----------
    alpha : torch.Tensor [num_aus, num_emotions]
        Dirichlet concentration parameters (pseudo-counts)
    total_counts : torch.Tensor [num_aus, 1]
        Total counts per AU (normalization factor)
    update_count : int
        Number of updates performed
    """

    def __init__(
        self,
        num_aus: int = 23,
        num_emotions: int = 6,
        prior_matrix: Optional[np.ndarray] = None,
        prior_strength: float = 100.0,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.prior_strength = prior_strength
        self.device_str = device

        # Initialize prior matrix
        if prior_matrix is None:
            # Uniform prior if not provided
            prior_matrix = np.ones((num_aus, num_emotions)) / num_emotions
        else:
            prior_matrix = np.array(prior_matrix)
            assert prior_matrix.shape == (num_aus, num_emotions), \
                f"Prior matrix shape mismatch: expected {(num_aus, num_emotions)}, got {prior_matrix.shape}"

            # Normalize rows to sum to 1
            row_sums = prior_matrix.sum(axis=1, keepdims=True)
            prior_matrix = prior_matrix / (row_sums + 1e-10)

        # Store original prior for regularization
        self.register_buffer(
            'prior',
            torch.tensor(prior_matrix, dtype=torch.float32, device=device)
        )

        # Initialize Dirichlet parameters (alpha = prior * strength)
        self.register_buffer(
            'alpha',
            self.prior * prior_strength
        )

        # Total counts (for normalization)
        self.register_buffer(
            'total_counts',
            torch.sum(self.alpha, dim=1, keepdim=True)
        )

        # Update statistics
        self.register_buffer(
            'update_count',
            torch.tensor(0, dtype=torch.long, device=device)
        )

        # Separate counters for seen/unseen updates
        self.register_buffer(
            'seen_update_count',
            torch.tensor(0, dtype=torch.long, device=device)
        )
        self.register_buffer(
            'unseen_update_count',
            torch.tensor(0, dtype=torch.long, device=device)
        )

    def get_probability(self) -> torch.Tensor:
        """
        Get current P(EMO|AU) probability matrix

        Returns:
        --------
        prob : torch.Tensor [num_aus, num_emotions]
            Probability distribution over emotions for each AU
            Each row sums to 1.0
        """
        return self.alpha / self.total_counts

    def forward(self, au_probs: torch.Tensor) -> torch.Tensor:
        """
        Predict emotion probabilities from AU probabilities

        P(EMO|sample) = Σ_i P(EMO|AU_i) * P(AU_i|sample)

        Parameters:
        -----------
        au_probs : torch.Tensor [batch_size, num_aus]
            AU activation probabilities

        Returns:
        --------
        emo_probs : torch.Tensor [batch_size, num_emotions]
            Emotion probabilities
        """
        prob_matrix = self.get_probability()  # [num_aus, num_emotions]
        emo_probs = torch.matmul(au_probs, prob_matrix)  # [batch, num_emotions]
        return emo_probs

    def update(
        self,
        au_probs: torch.Tensor,
        emo_labels: torch.Tensor,
        is_seen: bool = True,
        confidence: Optional[torch.Tensor] = None,
        seen_weight: float = 10.0,
        unseen_weight: float = 1.0,
        min_confidence: float = 0.8
    ) -> Dict[str, float]:
        """
        Bayesian update of AU-EMO matrix

        Parameters:
        -----------
        au_probs : torch.Tensor [batch_size, num_aus]
            AU activation probabilities
        emo_labels : torch.Tensor [batch_size]
            Emotion labels (integer indices)
        is_seen : bool
            Whether these are seen class samples
        confidence : torch.Tensor [batch_size], optional
            Confidence scores for each prediction
            If None, all samples have confidence 1.0
        seen_weight : float
            Update weight multiplier for seen classes
        unseen_weight : float
            Update weight multiplier for unseen classes
        min_confidence : float
            Minimum confidence threshold for unseen updates

        Returns:
        --------
        stats : dict
            Update statistics including number of samples updated
        """
        batch_size = au_probs.shape[0]

        # Default confidence
        if confidence is None:
            confidence = torch.ones(batch_size, device=au_probs.device)

        # Determine update weight
        base_weight = seen_weight if is_seen else unseen_weight

        # For unseen, filter by confidence
        if not is_seen:
            valid_mask = confidence >= min_confidence
            if not valid_mask.any():
                return {
                    'updated_samples': 0,
                    'skipped_samples': batch_size,
                    'avg_confidence': confidence.mean().item()
                }

            au_probs = au_probs[valid_mask]
            emo_labels = emo_labels[valid_mask]
            confidence = confidence[valid_mask]
            batch_size = au_probs.shape[0]
        else:
            valid_mask = torch.ones(batch_size, dtype=torch.bool, device=au_probs.device)

        # Create one-hot encoding for emotions
        emo_onehot = F.one_hot(emo_labels, num_classes=self.num_emotions).float()
        # [batch_size, num_emotions]

        # Weight each sample by base_weight and confidence
        sample_weights = base_weight * confidence  # [batch_size]

        # Compute update: Δα_ij = Σ_samples P(AU_i|sample) * I(emo=j) * weight
        # au_probs: [batch, num_aus]
        # emo_onehot: [batch, num_emotions]
        # sample_weights: [batch]

        # Weighted AU probabilities: [batch, num_aus]
        weighted_au = au_probs * sample_weights.unsqueeze(1)

        # Update matrix: [num_aus, num_emotions]
        update_matrix = torch.matmul(weighted_au.t(), emo_onehot)

        # Apply update
        self.alpha += update_matrix
        self.total_counts = torch.sum(self.alpha, dim=1, keepdim=True)

        # Update statistics
        self.update_count += batch_size
        if is_seen:
            self.seen_update_count += batch_size
        else:
            self.unseen_update_count += batch_size

        return {
            'updated_samples': batch_size,
            'skipped_samples': (~valid_mask).sum().item(),
            'avg_confidence': confidence.mean().item(),
            'update_norm': update_matrix.norm().item()
        }

    def regularize_to_prior(self, strength: float = 0.01) -> float:
        """
        Regularize matrix towards prior to prevent overfitting

        α ← (1 - strength) * α + strength * prior * prior_strength

        Parameters:
        -----------
        strength : float
            Regularization strength (0 = no regularization, 1 = full reset)

        Returns:
        --------
        divergence : float
            KL divergence from prior before regularization
        """
        current_prob = self.get_probability()

        # Compute KL divergence from prior
        kl_div = F.kl_div(
            torch.log(current_prob + 1e-10),
            self.prior,
            reduction='batchmean'
        ).item()

        # Apply regularization
        self.alpha = (1 - strength) * self.alpha + strength * self.prior * self.prior_strength
        self.total_counts = torch.sum(self.alpha, dim=1, keepdim=True)

        return kl_div

    def get_statistics(self) -> Dict:
        """Get update statistics"""
        return {
            'total_updates': self.update_count.item(),
            'seen_updates': self.seen_update_count.item(),
            'unseen_updates': self.unseen_update_count.item(),
            'avg_alpha': self.alpha.mean().item(),
            'matrix_entropy': self._compute_entropy(),
            'kl_from_prior': self._compute_kl_from_prior()
        }

    def _compute_entropy(self) -> float:
        """Compute average entropy of emotion distributions per AU"""
        prob = self.get_probability()
        entropy = -(prob * torch.log(prob + 1e-10)).sum(dim=1).mean()
        return entropy.item()

    def _compute_kl_from_prior(self) -> float:
        """Compute KL divergence from prior"""
        current_prob = self.get_probability()
        kl_div = F.kl_div(
            torch.log(current_prob + 1e-10),
            self.prior,
            reduction='batchmean'
        )
        return kl_div.item()

    def save(self, filepath: str):
        """Save matrix state to file"""
        state = {
            'alpha': self.alpha.cpu().numpy(),
            'prior': self.prior.cpu().numpy(),
            'update_count': self.update_count.item(),
            'seen_update_count': self.seen_update_count.item(),
            'unseen_update_count': self.unseen_update_count.item(),
            'num_aus': self.num_aus,
            'num_emotions': self.num_emotions,
            'prior_strength': self.prior_strength
        }

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as .npz for efficiency
        np.savez(filepath, **state)
        print(f"AU-EMO matrix saved to {filepath}")

    def load(self, filepath: str):
        """Load matrix state from file"""
        state = np.load(filepath)

        # Verify dimensions
        assert state['num_aus'] == self.num_aus
        assert state['num_emotions'] == self.num_emotions

        # Load state
        self.alpha.copy_(torch.tensor(state['alpha'], device=self.device_str))
        self.total_counts = torch.sum(self.alpha, dim=1, keepdim=True)
        self.update_count.copy_(torch.tensor(state['update_count'], device=self.device_str))
        self.seen_update_count.copy_(torch.tensor(state['seen_update_count'], device=self.device_str))
        self.unseen_update_count.copy_(torch.tensor(state['unseen_update_count'], device=self.device_str))

        print(f"AU-EMO matrix loaded from {filepath}")
        print(f"  Total updates: {self.update_count.item()}")
        print(f"  Seen updates: {self.seen_update_count.item()}")
        print(f"  Unseen updates: {self.unseen_update_count.item()}")

    def reset_to_prior(self):
        """Reset matrix to original prior"""
        self.alpha.copy_(self.prior * self.prior_strength)
        self.total_counts = torch.sum(self.alpha, dim=1, keepdim=True)
        self.update_count.fill_(0)
        self.seen_update_count.fill_(0)
        self.unseen_update_count.fill_(0)

    def visualize_matrix(self, au_names: Optional[list] = None,
                        emotion_names: Optional[list] = None) -> str:
        """
        Create a text visualization of the matrix

        Parameters:
        -----------
        au_names : list of str, optional
            Names of AUs for row labels
        emotion_names : list of str, optional
            Names of emotions for column labels

        Returns:
        --------
        viz : str
            Text representation of the matrix
        """
        import io

        prob = self.get_probability().cpu().numpy()

        # Default names
        if au_names is None:
            au_names = [f"AU{i}" for i in range(self.num_aus)]
        if emotion_names is None:
            emotion_names = [f"EMO{i}" for i in range(self.num_emotions)]

        # Create table
        output = io.StringIO()

        # Header
        output.write(f"{'AU':<10}")
        for emo_name in emotion_names:
            output.write(f"{emo_name:>10}")
        output.write("\n")
        output.write("-" * (10 + 10 * self.num_emotions) + "\n")

        # Rows
        for i, au_name in enumerate(au_names):
            output.write(f"{au_name:<10}")
            for j in range(self.num_emotions):
                output.write(f"{prob[i, j]:>10.4f}")
            output.write("\n")

        return output.getvalue()


def load_au_emo_prior(filepath: str) -> Tuple[np.ndarray, list, list]:
    """
    Load AU-EMO prior matrix from JSON file

    Expected format:
    {
        "au_names": ["AU1", "AU2", ...],
        "emotion_names": ["happy", "sad", ...],
        "prior_matrix": [[0.8, 0.1, ...], ...]
    }

    Returns:
    --------
    prior_matrix : np.ndarray
        Prior probability matrix
    au_names : list of str
        AU names
    emotion_names : list of str
        Emotion names
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    prior_matrix = np.array(data['prior_matrix'])
    au_names = data['au_names']
    emotion_names = data['emotion_names']

    return prior_matrix, au_names, emotion_names


# Example usage
if __name__ == "__main__":
    # Example: Create matrix with uniform prior
    num_aus = 23
    num_emotions = 6

    matrix = AUEMOMatrix(
        num_aus=num_aus,
        num_emotions=num_emotions,
        prior_strength=100.0,
        device='cpu'
    )

    print("Initial matrix:")
    print(matrix.visualize_matrix())

    # Simulate some updates
    batch_size = 32
    au_probs = torch.rand(batch_size, num_aus)
    emo_labels = torch.randint(0, num_emotions, (batch_size,))

    stats = matrix.update(au_probs, emo_labels, is_seen=True)
    print(f"\nAfter seen update: {stats}")

    # Predict emotions
    test_au = torch.rand(1, num_aus)
    emo_pred = matrix(test_au)
    print(f"\nPredicted emotion distribution: {emo_pred}")

    print(f"\nMatrix statistics: {matrix.get_statistics()}")
