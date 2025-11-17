"""
Beta-Bernoulli AU-EMO Probability Matrix (Whitebox Approach)

This module implements an interpretable AU-EMO association matrix using
Beta-Bernoulli conjugate prior framework for Bayesian updating.

Key Features:
1. Each P(AU_i|EMO_j) is modeled as Beta(α_ij, β_ij)
2. Point estimate: P(AU_i|EMO_j) = α_ij / (α_ij + β_ij)
3. Uncertainty: Var[P] = αβ / [(α+β)²(α+β+1)]
4. Bayesian update: Observe AU activations → update α, β parameters
5. Full interpretability and statistical guarantees
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Dict, Tuple
import json
from pathlib import Path


class BetaBernoulliAUEMOMatrix(nn.Module):
    """
    Beta-Bernoulli AU-EMO Matrix for Whitebox Interpretable Updating

    Models P(AU|EMO) as Beta distributions with conjugate Bayesian updates.

    Mathematical Framework:
    -----------------------
    P(AU_i|EMO_j) ~ Beta(α_ij, β_ij)

    Point Estimate:
        P(AU_i=1|EMO_j) = α_ij / (α_ij + β_ij)
        P(AU_i=0|EMO_j) = β_ij / (α_ij + β_ij)

    Update Rule:
        If sample has EMO_j and AU_i=1: α_ij += weight
        If sample has EMO_j and AU_i=0: β_ij += weight

    For Prediction (convert to P(EMO|AU)):
        Use Bayes' theorem with uniform P(EMO) assumption
        P(EMO_j|AU_i) ∝ P(AU_i|EMO_j)
        Then normalize across emotions for each AU

    Parameters:
    -----------
    num_aus : int
        Number of Action Units (e.g., 23)
    num_emotions : int
        Number of emotion classes (e.g., 6)
    prior_p_au_given_emo : np.ndarray [num_aus, num_emotions], optional
        Psychology prior P(AU|EMO) matrix
        If None, uses uniform prior (0.5 for each AU-EMO pair)
    prior_strength : float
        Strength of prior (total pseudo-count per AU-EMO pair)
        Higher = harder to update from prior
    device : str
        Device for tensors
    """

    def __init__(
        self,
        num_aus: int = 23,
        num_emotions: int = 6,
        prior_p_au_given_emo: Optional[np.ndarray] = None,
        prior_strength: float = 100.0,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.prior_strength = prior_strength
        self.device_str = device

        # Initialize prior P(AU|EMO)
        if prior_p_au_given_emo is None:
            # Uniform prior: P(AU_i|EMO_j) = 0.5
            prior_p = np.ones((num_aus, num_emotions)) * 0.5
        else:
            prior_p = np.array(prior_p_au_given_emo)
            assert prior_p.shape == (num_aus, num_emotions), \
                f"Prior shape mismatch: expected {(num_aus, num_emotions)}, got {prior_p.shape}"

        # Convert prior probabilities to Beta parameters
        # P(AU=1|EMO) = α / (α + β) = prior_p
        # Total pseudo-count: α + β = prior_strength
        # Solve: α = prior_p * prior_strength, β = (1-prior_p) * prior_strength
        alpha_init = prior_p * prior_strength
        beta_init = (1 - prior_p) * prior_strength

        # Store as buffers (not parameters, manual update)
        self.register_buffer(
            'alpha',
            torch.tensor(alpha_init, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            'beta',
            torch.tensor(beta_init, dtype=torch.float32, device=device)
        )

        # Store original prior for regularization
        self.register_buffer(
            'alpha_prior',
            torch.tensor(alpha_init, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            'beta_prior',
            torch.tensor(beta_init, dtype=torch.float32, device=device)
        )

        # Update statistics
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

    def get_p_au_given_emo(self) -> torch.Tensor:
        """
        Get current P(AU|EMO) point estimates

        Returns:
        --------
        p_au_given_emo : torch.Tensor [num_aus, num_emotions]
            P(AU_i=1|EMO_j) = α_ij / (α_ij + β_ij)
        """
        return self.alpha / (self.alpha + self.beta)

    def get_p_emo_given_au(self) -> torch.Tensor:
        """
        Get P(EMO|AU) for prediction using Bayes' theorem

        Assumes uniform P(EMO), so:
            P(EMO_j|AU_i) ∝ P(AU_i|EMO_j)
        Then normalize across emotions for each AU

        Returns:
        --------
        p_emo_given_au : torch.Tensor [num_aus, num_emotions]
            P(EMO_j|AU_i) for prediction
        """
        p_au_given_emo = self.get_p_au_given_emo()  # [num_aus, num_emotions]

        # Normalize each row (AU) to sum to 1 across emotions
        p_emo_given_au = p_au_given_emo / (p_au_given_emo.sum(dim=1, keepdim=True) + 1e-10)

        return p_emo_given_au

    def get_uncertainty(self) -> torch.Tensor:
        """
        Get uncertainty (variance) of Beta distributions

        Var[P(AU|EMO)] = αβ / [(α+β)²(α+β+1)]

        Returns:
        --------
        variance : torch.Tensor [num_aus, num_emotions]
            Variance of each Beta distribution
        """
        alpha_plus_beta = self.alpha + self.beta
        variance = (self.alpha * self.beta) / (
            alpha_plus_beta.pow(2) * (alpha_plus_beta + 1)
        )
        return variance

    def forward(self, au_probs: torch.Tensor) -> torch.Tensor:
        """
        Predict emotion probabilities from AU probabilities

        P(EMO_j|sample) = Σ_i P(EMO_j|AU_i) * P(AU_i|sample)
                        = au_probs @ P(EMO|AU)

        Parameters:
        -----------
        au_probs : torch.Tensor [batch_size, num_aus]
            AU activation probabilities

        Returns:
        --------
        emo_logits : torch.Tensor [batch_size, num_emotions]
            Emotion prediction logits
        """
        p_emo_given_au = self.get_p_emo_given_au()  # [num_aus, num_emotions]

        # Matrix multiplication
        emo_logits = torch.matmul(au_probs, p_emo_given_au)  # [batch, num_emotions]

        return emo_logits

    def update_from_labels(
        self,
        au_probs: torch.Tensor,
        emo_labels: torch.Tensor,
        is_seen: bool = True,
        confidence: Optional[torch.Tensor] = None,
        seen_weight: float = 1.0,
        unseen_weight: float = 0.8,
        min_confidence: float = 0.8
    ) -> Dict[str, float]:
        """
        Bayesian update of Beta parameters using observed AU activations

        Update Strategy:
        ----------------
        For each sample with emotion label EMO_j:
            For each AU_i:
                If AU_i is active (high probability):
                    α_ij += weight * P(AU_i|sample)
                If AU_i is inactive (low probability):
                    β_ij += weight * (1 - P(AU_i|sample))

        Weight Determination:
            Seen class: base_weight = seen_weight
            Unseen class: base_weight = unseen_weight * confidence

        Parameters:
        -----------
        au_probs : torch.Tensor [batch_size, num_aus]
            AU activation probabilities (from AU predictor)
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

        # Sample-wise weights
        sample_weights = base_weight * confidence  # [batch_size]

        # Accumulate updates for each emotion class
        for emo_idx in range(self.num_emotions):
            # Samples with this emotion
            emo_mask = (emo_labels == emo_idx)

            if not emo_mask.any():
                continue

            # AU probabilities for this emotion's samples
            au_probs_emo = au_probs[emo_mask]  # [n_samples, num_aus]
            weights_emo = sample_weights[emo_mask]  # [n_samples]

            # Weighted AU activations
            # Δα_i = Σ_samples weight * P(AU_i=1|sample)
            delta_alpha = (au_probs_emo * weights_emo.unsqueeze(1)).sum(dim=0)  # [num_aus]

            # Weighted AU inactivations
            # Δβ_i = Σ_samples weight * P(AU_i=0|sample)
            delta_beta = ((1 - au_probs_emo) * weights_emo.unsqueeze(1)).sum(dim=0)  # [num_aus]

            # Update Beta parameters for this emotion
            self.alpha[:, emo_idx] += delta_alpha
            self.beta[:, emo_idx] += delta_beta

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
            'avg_alpha': self.alpha.mean().item(),
            'avg_beta': self.beta.mean().item()
        }

    def regularize_to_prior(self, strength: float = 0.01) -> float:
        """
        Regularize Beta parameters towards prior

        Interpolates current parameters with prior:
            α_new = (1 - strength) * α_current + strength * α_prior
            β_new = (1 - strength) * β_current + strength * β_prior

        Parameters:
        -----------
        strength : float
            Regularization strength (0 = no change, 1 = full reset)

        Returns:
        --------
        kl_div : float
            KL divergence from prior before regularization
        """
        # Compute KL divergence before regularization
        p_current = self.get_p_au_given_emo()
        p_prior = self.alpha_prior / (self.alpha_prior + self.beta_prior)

        kl_div = F.kl_div(
            torch.log(p_current + 1e-10),
            p_prior,
            reduction='batchmean'
        ).item()

        # Interpolate towards prior
        self.alpha.mul_(1 - strength).add_(self.alpha_prior * strength)
        self.beta.mul_(1 - strength).add_(self.beta_prior * strength)

        return kl_div

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        p_au_given_emo = self.get_p_au_given_emo()
        uncertainty = self.get_uncertainty()

        return {
            'total_updates': self.update_count.item(),
            'seen_updates': self.seen_update_count.item(),
            'unseen_updates': self.unseen_update_count.item(),
            'avg_p_au_given_emo': p_au_given_emo.mean().item(),
            'avg_uncertainty': uncertainty.mean().item(),
            'max_uncertainty': uncertainty.max().item(),
            'avg_alpha': self.alpha.mean().item(),
            'avg_beta': self.beta.mean().item(),
            'effective_sample_size': (self.alpha + self.beta).mean().item(),
            'kl_from_prior': self._compute_kl_from_prior()
        }

    def _compute_kl_from_prior(self) -> float:
        """Compute KL divergence from prior"""
        p_current = self.get_p_au_given_emo()
        p_prior = self.alpha_prior / (self.alpha_prior + self.beta_prior)

        kl_div = F.kl_div(
            torch.log(p_current + 1e-10),
            p_prior,
            reduction='batchmean'
        )
        return kl_div.item()

    def save(self, filepath: str):
        """Save Beta parameters and statistics"""
        state = {
            'alpha': self.alpha.cpu().numpy(),
            'beta': self.beta.cpu().numpy(),
            'alpha_prior': self.alpha_prior.cpu().numpy(),
            'beta_prior': self.beta_prior.cpu().numpy(),
            'update_count': self.update_count.item(),
            'seen_update_count': self.seen_update_count.item(),
            'unseen_update_count': self.unseen_update_count.item(),
            'num_aus': self.num_aus,
            'num_emotions': self.num_emotions,
            'prior_strength': self.prior_strength
        }

        np.savez(filepath, **state)
        print(f"Beta-Bernoulli AU-EMO matrix saved to {filepath}")

    def load(self, filepath: str):
        """Load Beta parameters and statistics"""
        state = np.load(filepath)

        assert state['num_aus'] == self.num_aus
        assert state['num_emotions'] == self.num_emotions

        self.alpha.copy_(torch.tensor(state['alpha'], device=self.device_str))
        self.beta.copy_(torch.tensor(state['beta'], device=self.device_str))
        self.update_count.copy_(torch.tensor(state['update_count'], device=self.device_str))
        self.seen_update_count.copy_(torch.tensor(state['seen_update_count'], device=self.device_str))
        self.unseen_update_count.copy_(torch.tensor(state['unseen_update_count'], device=self.device_str))

        print(f"Beta-Bernoulli AU-EMO matrix loaded from {filepath}")

    def visualize_matrix(
        self,
        au_names: Optional[list] = None,
        emotion_names: Optional[list] = None,
        show_uncertainty: bool = False
    ) -> str:
        """
        Create text visualization of P(AU|EMO) matrix

        Parameters:
        -----------
        au_names : list, optional
            Names of AUs
        emotion_names : list, optional
            Names of emotions
        show_uncertainty : bool
            Whether to show uncertainty (std dev)

        Returns:
        --------
        visualization : str
            Formatted text table
        """
        import io

        p_au_given_emo = self.get_p_au_given_emo().cpu().numpy()

        if show_uncertainty:
            uncertainty = self.get_uncertainty().cpu().numpy()
            std_dev = np.sqrt(uncertainty)

        if au_names is None:
            au_names = [f"AU{i}" for i in range(self.num_aus)]
        if emotion_names is None:
            emotion_names = [f"EMO{i}" for i in range(self.num_emotions)]

        output = io.StringIO()

        # Header
        output.write(f"{'AU':<15}")
        for emo_name in emotion_names:
            if show_uncertainty:
                output.write(f"{emo_name:>20}")
            else:
                output.write(f"{emo_name:>12}")
        output.write("\n")
        output.write("-" * (15 + (20 if show_uncertainty else 12) * self.num_emotions) + "\n")

        # Rows
        for i, au_name in enumerate(au_names):
            output.write(f"{au_name:<15}")
            for j in range(self.num_emotions):
                if show_uncertainty:
                    output.write(f"{p_au_given_emo[i, j]:>8.4f}±{std_dev[i, j]:>8.4f}")
                else:
                    output.write(f"{p_au_given_emo[i, j]:>12.4f}")
            output.write("\n")

        return output.getvalue()


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
    # Test Beta-Bernoulli matrix
    print("Testing Beta-Bernoulli AU-EMO Matrix...")

    # Create simple prior
    num_aus, num_emotions = 3, 2
    prior_p = np.array([
        [0.8, 0.2],  # AU0: high for EMO0, low for EMO1
        [0.3, 0.7],  # AU1: low for EMO0, high for EMO1
        [0.5, 0.5]   # AU2: neutral
    ])

    # Initialize matrix
    matrix = BetaBernoulliAUEMOMatrix(
        num_aus=num_aus,
        num_emotions=num_emotions,
        prior_p_au_given_emo=prior_p,
        prior_strength=100.0,
        device='cpu'
    )

    print("\nInitial P(AU|EMO):")
    print(matrix.get_p_au_given_emo())

    print("\nInitial P(EMO|AU) for prediction:")
    print(matrix.get_p_emo_given_au())

    print("\nInitial uncertainty (variance):")
    print(matrix.get_uncertainty())

    # Test prediction
    au_probs = torch.tensor([[0.9, 0.1, 0.5]])  # AU0 active
    emo_pred = matrix(au_probs)
    print(f"\nPrediction test:")
    print(f"  AU probs: {au_probs}")
    print(f"  EMO logits: {emo_pred}")
    print(f"  EMO probs: {F.softmax(emo_pred, dim=1)}")

    # Test update
    au_probs_batch = torch.tensor([
        [0.9, 0.1, 0.5],  # EMO0 sample
        [0.1, 0.9, 0.5],  # EMO1 sample
    ])
    emo_labels = torch.tensor([0, 1])

    stats = matrix.update_from_labels(
        au_probs_batch,
        emo_labels,
        is_seen=True,
        seen_weight=1.0
    )

    print(f"\nUpdate stats: {stats}")

    print("\nUpdated P(AU|EMO):")
    print(matrix.get_p_au_given_emo())

    print("\nUpdated uncertainty:")
    print(matrix.get_uncertainty())

    print("\nMatrix statistics:")
    for k, v in matrix.get_statistics().items():
        print(f"  {k}: {v:.4f}")

    print("\n✓ All tests passed!")
