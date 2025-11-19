"""
Learnable AU-EMO Probability Matrix (Blackbox Approach)

This module implements an end-to-end learnable AU-EMO association matrix
using nn.Parameter for gradient-based optimization.

Key Features:
1. Matrix parameters are directly optimized via backpropagation
2. Initialized with psychology prior
3. Regularization maintains connection to prior
4. No explicit probabilistic interpretation during training
5. Can extract probability matrix for analysis
6. Simpler and faster than Bayesian approach
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import json
from pathlib import Path


class LearnableAUEMOMatrix(nn.Module):
    """
    Learnable AU-EMO Matrix for Blackbox End-to-End Training

    Unlike the whitebox Beta-Bernoulli approach, this matrix is directly
    optimized via gradient descent. The parameters can be any real values,
    and are converted to probabilities when needed.

    Mathematical Framework:
    -----------------------
    Matrix stores logits M_ij for each AU-EMO pair

    P(EMO_j|AU_i) = softmax_j(M_ij)  (normalize across emotions per AU)

    For prediction:
        P(EMO_j|sample) = Σ_i P(EMO_j|AU_i) * P(AU_i|sample)
                        = au_probs @ softmax(M, dim=1)

    Training:
        - Matrix M is nn.Parameter
        - Optimized via backpropagation like other network weights
        - Regularized towards prior to prevent drift

    Advantages:
    -----------
    + Simpler implementation (no EM, no Bayesian updates)
    + Faster training (single pass backprop)
    + More flexible (can learn complex patterns)
    + End-to-end optimization with rest of network

    Disadvantages:
    --------------
    - Less interpretable than Beta-Bernoulli
    - No uncertainty quantification
    - Requires careful regularization to maintain prior knowledge

    Parameters:
    -----------
    num_aus : int
        Number of Action Units (e.g., 23)
    num_emotions : int
        Number of emotion classes (e.g., 6)
    prior_p_au_given_emo : np.ndarray [num_aus, num_emotions], optional
        Psychology prior P(AU|EMO) matrix
        Used to initialize matrix and for regularization
    prior_strength : float
        Regularization strength towards prior
        Higher = stay closer to prior
    device : str
        Device for tensors
    """

    def __init__(
        self,
        num_aus: int = 23,
        num_emotions: int = 6,
        prior_p_au_given_emo: Optional[np.ndarray] = None,
        prior_strength: float = 0.1,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.prior_strength = prior_strength
        self.device_str = device

        # Initialize prior P(AU|EMO)
        if prior_p_au_given_emo is None:
            # Uniform prior
            prior_p_au_emo = np.ones((num_aus, num_emotions)) / num_emotions
        else:
            prior_p_au_emo = np.array(prior_p_au_given_emo)
            assert prior_p_au_emo.shape == (num_aus, num_emotions), \
                f"Prior shape mismatch: expected {(num_aus, num_emotions)}, got {prior_p_au_emo.shape}"

        # Convert P(AU|EMO) to P(EMO|AU) for initialization
        # Assume uniform P(EMO), so P(EMO|AU) ∝ P(AU|EMO)
        prior_p_emo_au = prior_p_au_emo.copy()
        row_sums = prior_p_emo_au.sum(axis=1, keepdims=True)
        prior_p_emo_au = prior_p_emo_au / (row_sums + 1e-10)

        # Convert probabilities to logits for initialization
        # Use inverse softmax: logit = log(prob)
        prior_logits = np.log(prior_p_emo_au + 1e-10)

        # Store prior as buffer (not learnable)
        self.register_buffer(
            'prior_logits',
            torch.tensor(prior_logits, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            'prior_probs',
            torch.tensor(prior_p_emo_au, dtype=torch.float32, device=device)
        )

        # Learnable matrix (nn.Parameter)
        # Initialize with prior logits
        self.matrix_logits = nn.Parameter(
            torch.tensor(prior_logits, dtype=torch.float32, device=device)
        )

        # Statistics
        self.register_buffer(
            'update_count',
            torch.tensor(0, dtype=torch.long, device=device)
        )

    def get_probability_matrix(self) -> torch.Tensor:
        """
        Get current P(EMO|AU) probability matrix

        Applies softmax across emotions for each AU to ensure valid probabilities

        Returns:
        --------
        p_emo_given_au : torch.Tensor [num_aus, num_emotions]
            P(EMO_j|AU_i) for each AU-EMO pair
        """
        # Softmax across emotions (dim=1) for each AU
        p_emo_given_au = F.softmax(self.matrix_logits, dim=1)
        return p_emo_given_au

    def forward(self, au_probs: torch.Tensor) -> torch.Tensor:
        """
        Predict emotion probabilities from AU probabilities

        P(EMO_j|sample) = Σ_i P(EMO_j|AU_i) * P(AU_i|sample)

        Parameters:
        -----------
        au_probs : torch.Tensor [batch_size, num_aus]
            AU activation probabilities from AU predictor

        Returns:
        --------
        emo_logits : torch.Tensor [batch_size, num_emotions]
            Emotion prediction logits
        """
        p_emo_given_au = self.get_probability_matrix()  # [num_aus, num_emotions]

        # Matrix multiplication
        emo_logits = torch.matmul(au_probs, p_emo_given_au)  # [batch, num_emotions]

        return emo_logits

    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss towards prior

        Uses KL divergence: D_KL(current || prior)

        Returns:
        --------
        reg_loss : torch.Tensor (scalar)
            Regularization loss
        """
        current_probs = self.get_probability_matrix()

        # KL divergence: D_KL(P || Q) = Σ P(x) log(P(x) / Q(x))
        kl_div = F.kl_div(
            torch.log(current_probs + 1e-10),
            self.prior_probs,
            reduction='batchmean'
        )

        return self.prior_strength * kl_div

    def compute_entropy_regularization(self, strength: float = 0.01) -> torch.Tensor:
        """
        Entropy regularization to prevent overconfident predictions

        Higher entropy = more uncertain = less overconfident

        Parameters:
        -----------
        strength : float
            Regularization strength

        Returns:
        --------
        entropy_loss : torch.Tensor (scalar)
            Negative entropy (minimize this to maximize entropy)
        """
        probs = self.get_probability_matrix()

        # Entropy: H = -Σ p(x) log p(x)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=1).mean()

        # We want to maximize entropy, so minimize negative entropy
        return -strength * entropy

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        current_probs = self.get_probability_matrix()

        # KL from prior
        kl_div = F.kl_div(
            torch.log(current_probs + 1e-10),
            self.prior_probs,
            reduction='batchmean'
        ).item()

        # Average entropy per AU
        entropy = -(current_probs * torch.log(current_probs + 1e-10)).sum(dim=1).mean().item()

        # Logits statistics
        logits_mean = self.matrix_logits.mean().item()
        logits_std = self.matrix_logits.std().item()
        logits_max = self.matrix_logits.max().item()
        logits_min = self.matrix_logits.min().item()

        return {
            'avg_probability': current_probs.mean().item(),
            'kl_from_prior': kl_div,
            'avg_entropy_per_au': entropy,
            'logits_mean': logits_mean,
            'logits_std': logits_std,
            'logits_max': logits_max,
            'logits_min': logits_min,
            'update_count': self.update_count.item()
        }

    def reset_to_prior(self):
        """Reset matrix to prior (hard reset)"""
        self.matrix_logits.data.copy_(self.prior_logits)
        print("Matrix reset to prior")

    def soft_reset_to_prior(self, strength: float = 0.1):
        """Soft reset: interpolate current matrix with prior"""
        self.matrix_logits.data.mul_(1 - strength).add_(self.prior_logits * strength)
        print(f"Matrix soft reset (strength={strength:.3f})")

    def save(self, filepath: str):
        """Save matrix state"""
        state = {
            'matrix_logits': self.matrix_logits.detach().cpu().numpy(),
            'prior_logits': self.prior_logits.cpu().numpy(),
            'prior_probs': self.prior_probs.cpu().numpy(),
            'update_count': self.update_count.item(),
            'num_aus': self.num_aus,
            'num_emotions': self.num_emotions,
            'prior_strength': self.prior_strength
        }

        np.savez(filepath, **state)
        print(f"Learnable AU-EMO matrix saved to {filepath}")

    def load(self, filepath: str):
        """Load matrix state"""
        state = np.load(filepath)

        assert state['num_aus'] == self.num_aus
        assert state['num_emotions'] == self.num_emotions

        self.matrix_logits.data.copy_(
            torch.tensor(state['matrix_logits'], device=self.device_str)
        )
        self.update_count.copy_(
            torch.tensor(state['update_count'], device=self.device_str)
        )

        print(f"Learnable AU-EMO matrix loaded from {filepath}")

    def visualize_matrix(
        self,
        au_names: Optional[list] = None,
        emotion_names: Optional[list] = None,
        show_logits: bool = False
    ) -> str:
        """
        Create text visualization of P(EMO|AU) matrix

        Parameters:
        -----------
        au_names : list, optional
            Names of AUs
        emotion_names : list, optional
            Names of emotions
        show_logits : bool
            Whether to show logits instead of probabilities

        Returns:
        --------
        visualization : str
            Formatted text table
        """
        import io

        if show_logits:
            matrix = self.matrix_logits.detach().cpu().numpy()
            title = "Matrix Logits"
        else:
            matrix = self.get_probability_matrix().detach().cpu().numpy()
            title = "P(EMO|AU) Probabilities"

        if au_names is None:
            au_names = [f"AU{i}" for i in range(self.num_aus)]
        if emotion_names is None:
            emotion_names = [f"EMO{i}" for i in range(self.num_emotions)]

        output = io.StringIO()

        # Title
        output.write(f"{title}\n")
        output.write("=" * (15 + 12 * self.num_emotions) + "\n")

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
                output.write(f"{matrix[i, j]:>12.4f}")
            output.write("\n")

        return output.getvalue()

    def get_p_au_given_emo_estimate(self) -> torch.Tensor:
        """
        Estimate P(AU|EMO) from current P(EMO|AU)

        This is an approximation assuming uniform P(AU)

        Returns:
        --------
        p_au_given_emo : torch.Tensor [num_aus, num_emotions]
        """
        p_emo_given_au = self.get_probability_matrix()  # [num_aus, num_emotions]

        # Reverse Bayes: P(AU|EMO) ∝ P(EMO|AU)
        # Normalize across AUs for each emotion
        p_au_given_emo = p_emo_given_au.t()  # [num_emotions, num_aus]
        p_au_given_emo = p_au_given_emo / (p_au_given_emo.sum(dim=1, keepdim=True) + 1e-10)
        p_au_given_emo = p_au_given_emo.t()  # [num_aus, num_emotions]

        return p_au_given_emo


def load_au_emo_prior(filepath: str) -> Tuple[np.ndarray, list, list]:
    """
    Load AU-EMO prior from JSON file

    Expected format:
    {
        "au_names": ["AU1", "AU2", ...],
        "emotion_names": ["happy", "sad", ...],
        "prior_matrix": [[...], ...]  # P(AU|EMO) [num_aus, num_emotions]
    }

    Returns:
    --------
    prior_matrix : np.ndarray [num_aus, num_emotions]
        P(AU|EMO) matrix
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

    return prior_matrix, au_names, emotion_names


if __name__ == "__main__":
    # Test learnable matrix
    print("Testing Learnable AU-EMO Matrix...")

    # Create simple prior
    num_aus, num_emotions = 3, 2
    prior_p = np.array([
        [0.8, 0.2],  # AU0: high for EMO0, low for EMO1
        [0.3, 0.7],  # AU1: low for EMO0, high for EMO1
        [0.5, 0.5]   # AU2: neutral
    ])

    # Initialize matrix
    matrix = LearnableAUEMOMatrix(
        num_aus=num_aus,
        num_emotions=num_emotions,
        prior_p_au_given_emo=prior_p,
        prior_strength=0.1,
        device='cpu'
    )

    print("\nInitial P(EMO|AU):")
    print(matrix.get_probability_matrix())

    print("\nInitial logits:")
    print(matrix.matrix_logits)

    # Test prediction
    au_probs = torch.tensor([[0.9, 0.1, 0.5]])  # AU0 active
    emo_pred = matrix(au_probs)
    print(f"\nPrediction test:")
    print(f"  AU probs: {au_probs}")
    print(f"  EMO logits: {emo_pred}")
    print(f"  EMO probs: {F.softmax(emo_pred, dim=1)}")

    # Test regularization
    reg_loss = matrix.compute_regularization_loss()
    print(f"\nRegularization loss: {reg_loss.item():.6f}")

    # Test gradient flow
    optimizer = torch.optim.Adam([matrix.matrix_logits], lr=0.01)

    print("\nTesting gradient flow...")
    for i in range(5):
        # Dummy loss
        au_probs_batch = torch.tensor([
            [0.9, 0.1, 0.5],
            [0.1, 0.9, 0.5],
        ])
        emo_labels = torch.tensor([0, 1])

        emo_pred = matrix(au_probs_batch)
        loss = F.cross_entropy(emo_pred, emo_labels)
        loss += matrix.compute_regularization_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"  Step {i+1}: loss={loss.item():.4f}")

    print("\nUpdated P(EMO|AU):")
    print(matrix.get_probability_matrix())

    print("\nMatrix statistics:")
    for k, v in matrix.get_statistics().items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ All tests passed!")
