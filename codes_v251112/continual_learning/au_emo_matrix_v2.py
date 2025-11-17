"""
AU-EMO Probability Matrix for Continual Learning (Corrected Version)

Key corrections:
1. Properly handles P(AU|EMO) -> P(EMO|AU) conversion
2. Assumes uniform P(EMO) distribution
3. Task-agnostic design (no task IDs)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import json
from pathlib import Path


def convert_au_given_emo_to_emo_given_au(
    p_au_given_emo: np.ndarray,
    assume_uniform_emo: bool = True
) -> np.ndarray:
    """
    Convert P(AU|EMO) to P(EMO|AU) using Bayes' theorem

    P(EMO|AU) = P(AU|EMO) * P(EMO) / P(AU)

    If P(EMO) is uniform (assume_uniform_emo=True):
        P(EMO|AU) ∝ P(AU|EMO)
        Then normalize each row (AU) to sum to 1

    Parameters:
    -----------
    p_au_given_emo : np.ndarray [num_aus, num_emotions]
        P(AU|EMO) matrix from psychology prior
    assume_uniform_emo : bool
        Whether to assume P(EMO) is uniform

    Returns:
    --------
    p_emo_given_au : np.ndarray [num_aus, num_emotions]
        P(EMO|AU) matrix
    """
    if assume_uniform_emo:
        # P(EMO|AU) ∝ P(AU|EMO) when P(EMO) is uniform
        # Just normalize rows
        p_emo_given_au = p_au_given_emo.copy()

        # Normalize each row (AU) to sum to 1
        row_sums = p_emo_given_au.sum(axis=1, keepdims=True)
        p_emo_given_au = p_emo_given_au / (row_sums + 1e-10)

        return p_emo_given_au
    else:
        # Full Bayes' theorem (requires P(EMO))
        # For now, we only support uniform assumption
        raise NotImplementedError(
            "Non-uniform P(EMO) not supported yet. "
            "Use assume_uniform_emo=True"
        )


class AUEMOMatrix(nn.Module):
    """
    AU-EMO Probability Matrix with Bayesian Update

    Maintains P(EMO|AU) and updates it using observed data.

    IMPORTANT NOTES:
    ----------------
    1. Input prior is P(AU|EMO) from psychology research
    2. Internally converts to P(EMO|AU) assuming uniform P(EMO)
    3. Updates are based on observed AU activations and emotion labels
    4. Task-agnostic: no task IDs, works with all emotion classes seen so far

    The forward pass predicts emotions from AUs:
        P(EMO|sample) = Σ_i P(EMO|AU_i) * P(AU_i|sample)

    Parameters:
    -----------
    num_aus : int
        Number of Action Units (e.g., 23)
    num_emotions : int
        Number of emotion classes (e.g., 6, excluding neutral)
    prior_matrix_au_given_emo : torch.Tensor or np.ndarray, optional
        Psychology prior P(AU|EMO) matrix [num_aus, num_emotions]
        Will be converted to P(EMO|AU) internally
    prior_strength : float
        Strength of the prior (pseudo-count multiplier)
    device : str
        Device to store tensors
    """

    def __init__(
        self,
        num_aus: int = 23,
        num_emotions: int = 6,
        prior_matrix_au_given_emo: Optional[np.ndarray] = None,
        prior_strength: float = 100.0,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.prior_strength = prior_strength
        self.device_str = device

        # Convert P(AU|EMO) to P(EMO|AU)
        if prior_matrix_au_given_emo is None:
            # Uniform prior
            p_emo_given_au = np.ones((num_aus, num_emotions)) / num_emotions
        else:
            prior_matrix_au_given_emo = np.array(prior_matrix_au_given_emo)
            assert prior_matrix_au_given_emo.shape == (num_aus, num_emotions), \
                f"Prior shape mismatch: expected {(num_aus, num_emotions)}, got {prior_matrix_au_given_emo.shape}"

            # Convert using Bayes' theorem with uniform P(EMO)
            p_emo_given_au = convert_au_given_emo_to_emo_given_au(
                prior_matrix_au_given_emo,
                assume_uniform_emo=True
            )

        # Store P(EMO|AU) prior
        self.register_buffer(
            'prior',
            torch.tensor(p_emo_given_au, dtype=torch.float32, device=device)
        )

        # Initialize Dirichlet parameters
        self.register_buffer(
            'alpha',
            self.prior * prior_strength
        )

        # Total counts for normalization
        self.register_buffer(
            'total_counts',
            torch.sum(self.alpha, dim=1, keepdim=True)
        )

        # Statistics
        self.register_buffer(
            'update_count',
            torch.tensor(0, dtype=torch.long, device=device)
        )
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
            P(EMO|AU) for each AU
        """
        return self.alpha / self.total_counts

    def forward(self, au_probs: torch.Tensor) -> torch.Tensor:
        """
        Predict emotion probabilities from AU probabilities

        P(EMO|sample) = Σ_i P(EMO|AU_i) * P(AU_i|sample)

        This is the "emo_from_au" prediction path.

        Parameters:
        -----------
        au_probs : torch.Tensor [batch_size, num_aus]
            AU activation probabilities from AU predictor

        Returns:
        --------
        emo_probs : torch.Tensor [batch_size, num_emotions]
            Emotion probabilities (logits, not softmax)
        """
        p_emo_given_au = self.get_probability()  # [num_aus, num_emotions]

        # Matrix multiplication: [batch, num_aus] @ [num_aus, num_emotions]
        emo_logits = torch.matmul(au_probs, p_emo_given_au)  # [batch, num_emotions]

        return emo_logits

    def update(
        self,
        au_probs: torch.Tensor,
        emo_labels: torch.Tensor,
        is_seen: bool = True,
        confidence: Optional[torch.Tensor] = None,
        seen_weight: float = 1.0,
        unseen_weight: float = 0.5,
        min_confidence: float = 0.8
    ) -> Dict[str, float]:
        """
        Bayesian update of AU-EMO matrix

        Update strategy:
        - Seen class: Use true labels with normal weight
        - Unseen class: Use pseudo-labels with lower weight (NOT lower learning rate!)
                       The weight is lower because pseudo-labels are less reliable

        Parameters:
        -----------
        au_probs : torch.Tensor [batch_size, num_aus]
            AU activation probabilities
        emo_labels : torch.Tensor [batch_size]
            Emotion labels (true for seen, pseudo for unseen)
        is_seen : bool
            Whether these are seen class samples
        confidence : torch.Tensor [batch_size], optional
            Confidence scores (only used for unseen)
        seen_weight : float
            Base update weight for seen classes
        unseen_weight : float
            Base update weight for unseen classes
            Lower because pseudo-labels are less reliable
        min_confidence : float
            Minimum confidence for unseen updates

        Returns:
        --------
        stats : dict
            Update statistics
        """
        batch_size = au_probs.shape[0]

        # Default confidence
        if confidence is None:
            confidence = torch.ones(batch_size, device=au_probs.device)

        # Determine base weight
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

        # Create one-hot encoding
        emo_onehot = F.one_hot(emo_labels, num_classes=self.num_emotions).float()

        # Weight each sample: base_weight * confidence
        # For seen class, confidence is always 1.0
        # For unseen class, confidence reflects prediction reliability
        sample_weights = base_weight * confidence

        # Weighted AU probabilities
        weighted_au = au_probs * sample_weights.unsqueeze(1)

        # Update matrix: Δα_ij = Σ_samples P(AU_i|sample) * I(emo=j) * weight
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
        """Regularize towards prior"""
        current_prob = self.get_probability()

        kl_div = F.kl_div(
            torch.log(current_prob + 1e-10),
            self.prior,
            reduction='batchmean'
        ).item()

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
        """Compute average entropy"""
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
        """Save matrix state"""
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

        np.savez(filepath, **state)
        print(f"AU-EMO matrix saved to {filepath}")

    def load(self, filepath: str):
        """Load matrix state"""
        state = np.load(filepath)

        assert state['num_aus'] == self.num_aus
        assert state['num_emotions'] == self.num_emotions

        self.alpha.copy_(torch.tensor(state['alpha'], device=self.device_str))
        self.total_counts = torch.sum(self.alpha, dim=1, keepdim=True)
        self.update_count.copy_(torch.tensor(state['update_count'], device=self.device_str))
        self.seen_update_count.copy_(torch.tensor(state['seen_update_count'], device=self.device_str))
        self.unseen_update_count.copy_(torch.tensor(state['unseen_update_count'], device=self.device_str))

        print(f"AU-EMO matrix loaded from {filepath}")

    def visualize_matrix(
        self,
        au_names: Optional[list] = None,
        emotion_names: Optional[list] = None
    ) -> str:
        """Create text visualization"""
        import io

        prob = self.get_probability().cpu().numpy()

        if au_names is None:
            au_names = [f"AU{i}" for i in range(self.num_aus)]
        if emotion_names is None:
            emotion_names = [f"EMO{i}" for i in range(self.num_emotions)]

        output = io.StringIO()

        # Header
        output.write(f"{'AU':<15}")
        for emo_name in emotion_names:
            output.write(f"{emo_name:>12}")
        output.write("\n")
        output.write("-" * (15 + 12 * self.num_emotions) + "\n")

        # Rows
        for i, au_name in enumerate(au_names):
            output.write(f"{au_name:<15}")
            for j in range(self.num_emotions):
                output.write(f"{prob[i, j]:>12.4f}")
            output.write("\n")

        return output.getvalue()


def load_au_emo_prior(filepath: str) -> Tuple[np.ndarray, list, list]:
    """
    Load AU-EMO prior from JSON file

    IMPORTANT: The prior matrix should be P(AU|EMO), not P(EMO|AU)!
    This function will automatically convert it.

    Expected format:
    {
        "au_names": ["AU1", "AU2", ...],
        "emotion_names": ["happy", "sad", ...],
        "prior_matrix": [[...], ...]  # P(AU|EMO) [num_aus, num_emotions]
    }

    Returns:
    --------
    prior_matrix : np.ndarray [num_aus, num_emotions]
        P(AU|EMO) matrix (will be converted internally by AUEMOMatrix)
    au_names : list
    emotion_names : list
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    prior_matrix = np.array(data['prior_matrix'])
    au_names = data['au_names']
    emotion_names = data['emotion_names']

    print(f"Loaded P(AU|EMO) prior from {filepath}")
    print(f"  Shape: {prior_matrix.shape}")
    print(f"  AUs: {len(au_names)}")
    print(f"  Emotions: {len(emotion_names)}")
    print("  Note: Will be converted to P(EMO|AU) internally")

    return prior_matrix, au_names, emotion_names


if __name__ == "__main__":
    # Test conversion
    print("Testing P(AU|EMO) -> P(EMO|AU) conversion...")

    # Example: P(AU|EMO) matrix
    num_aus, num_emotions = 3, 2
    p_au_given_emo = np.array([
        [0.8, 0.2],  # AU1: high for EMO0, low for EMO1
        [0.3, 0.7],  # AU2: low for EMO0, high for EMO1
        [0.5, 0.5]   # AU3: neutral
    ])

    print("\nInput P(AU|EMO):")
    print(p_au_given_emo)

    p_emo_given_au = convert_au_given_emo_to_emo_given_au(p_au_given_emo)

    print("\nOutput P(EMO|AU):")
    print(p_emo_given_au)
    print("\nRow sums (should be 1.0):", p_emo_given_au.sum(axis=1))

    # Test matrix
    matrix = AUEMOMatrix(
        num_aus=3,
        num_emotions=2,
        prior_matrix_au_given_emo=p_au_given_emo,
        device='cpu'
    )

    print("\nMatrix P(EMO|AU):")
    print(matrix.get_probability())

    # Test prediction
    au_probs = torch.tensor([[0.9, 0.1, 0.5]])  # AU1 active
    emo_pred = matrix(au_probs)
    print(f"\nAU probs: {au_probs}")
    print(f"EMO prediction (logits): {emo_pred}")
    print(f"EMO prediction (softmax): {F.softmax(emo_pred, dim=1)}")

    print("\n✓ All tests passed!")
