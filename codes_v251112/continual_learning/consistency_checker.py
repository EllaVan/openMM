"""
Multimodal Consistency Checker for Unseen Class Pseudo-labeling

This module implements various consistency strategies to filter
reliable pseudo-labels for unseen emotion classes in continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from enum import Enum
import numpy as np


class ConsistencyStrategy(Enum):
    """Consistency checking strategies"""
    ALL_AGREE = "all_agree"  # All 4 predictions must match
    MAJORITY = "majority"  # At least 3/4 must match
    WEIGHTED_VOTE = "weighted_vote"  # Weighted voting by confidence
    ENTROPY_THRESHOLD = "entropy_threshold"  # Low prediction entropy
    COMBINED = "combined"  # Combination of strategies


class MultimodalConsistencyChecker:
    """
    Multimodal Consistency Checker

    Checks consistency across different modalities and fusion predictions
    to determine reliable pseudo-labels for unseen classes.

    The checker evaluates predictions from:
    1. Text-only branch
    2. Audio-only branch
    3. Video-only branch
    4. Fused multimodal branch

    Parameters:
    -----------
    model : AUEmotionNetwork
        The main network
    strategy : ConsistencyStrategy
        Consistency checking strategy
    min_confidence : float
        Minimum confidence threshold
    entropy_threshold : float
        Maximum entropy threshold (for entropy-based checking)
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: ConsistencyStrategy = ConsistencyStrategy.MAJORITY,
        min_confidence: float = 0.7,
        entropy_threshold: float = 0.5
    ):
        self.model = model
        self.strategy = strategy
        self.min_confidence = min_confidence
        self.entropy_threshold = entropy_threshold

        # Statistics
        self.total_checked = 0
        self.total_consistent = 0
        self.consistency_by_class = {}

    def check_consistency(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Check multimodal consistency

        Args:
            text_features: [batch_size, T, text_dim]
            audio_features: [batch_size, T, audio_dim]
            video_features: [batch_size, T, video_dim]
            masks: [batch_size, T]

        Returns:
            dict: {
                'is_consistent': [batch_size] bool tensor,
                'consensus_label': [batch_size] int tensor,
                'confidence': [batch_size] float tensor,
                'predictions': {
                    'text': [batch_size],
                    'audio': [batch_size],
                    'video': [batch_size],
                    'fused': [batch_size]
                },
                'probabilities': {
                    'text': [batch_size, num_classes],
                    'audio': [batch_size, num_classes],
                    'video': [batch_size, num_classes],
                    'fused': [batch_size, num_classes]
                }
            }
        """
        batch_size = text_features.shape[0]
        device = text_features.device

        with torch.no_grad():
            # Get predictions from each modality
            predictions = {}
            probabilities = {}

            # 1. Text-only prediction
            text_pred, text_prob = self._predict_single_modality(
                text_features, masks, 'text'
            )
            predictions['text'] = text_pred
            probabilities['text'] = text_prob

            # 2. Audio-only prediction
            audio_pred, audio_prob = self._predict_single_modality(
                audio_features, masks, 'audio'
            )
            predictions['audio'] = audio_pred
            probabilities['audio'] = audio_prob

            # 3. Video-only prediction
            video_pred, video_prob = self._predict_single_modality(
                video_features, masks, 'video'
            )
            predictions['video'] = video_pred
            probabilities['video'] = video_prob

            # 4. Fused multimodal prediction
            fused_pred, fused_prob = self._predict_fused(
                text_features, audio_features, video_features, masks
            )
            predictions['fused'] = fused_pred
            probabilities['fused'] = fused_prob

        # Check consistency based on strategy
        if self.strategy == ConsistencyStrategy.ALL_AGREE:
            is_consistent, consensus_label, confidence = self._check_all_agree(
                predictions, probabilities
            )
        elif self.strategy == ConsistencyStrategy.MAJORITY:
            is_consistent, consensus_label, confidence = self._check_majority(
                predictions, probabilities
            )
        elif self.strategy == ConsistencyStrategy.WEIGHTED_VOTE:
            is_consistent, consensus_label, confidence = self._check_weighted_vote(
                predictions, probabilities
            )
        elif self.strategy == ConsistencyStrategy.ENTROPY_THRESHOLD:
            is_consistent, consensus_label, confidence = self._check_entropy(
                probabilities
            )
        else:  # COMBINED
            is_consistent, consensus_label, confidence = self._check_combined(
                predictions, probabilities
            )

        # Update statistics
        self.total_checked += batch_size
        self.total_consistent += is_consistent.sum().item()

        return {
            'is_consistent': is_consistent,
            'consensus_label': consensus_label,
            'confidence': confidence,
            'predictions': predictions,
            'probabilities': probabilities
        }

    def _predict_single_modality(
        self,
        features: torch.Tensor,
        masks: Optional[torch.Tensor],
        modality: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using single modality

        Args:
            features: [batch_size, T, feature_dim]
            masks: [batch_size, T]
            modality: 'text', 'audio', or 'video'

        Returns:
            predictions: [batch_size]
            probabilities: [batch_size, num_classes]
        """
        # Create zero features for other modalities
        batch_size, T, _ = features.shape
        device = features.device

        if modality == 'text':
            text_feat = features
            audio_feat = torch.zeros_like(features)
            video_feat = torch.zeros_like(features)
        elif modality == 'audio':
            text_feat = torch.zeros_like(features)
            audio_feat = features
            video_feat = torch.zeros_like(features)
        else:  # video
            text_feat = torch.zeros_like(features)
            audio_feat = torch.zeros_like(features)
            video_feat = features

        # Forward pass
        output = self.model(text_feat, audio_feat, video_feat, masks)

        # Get predictions from AU path (for unseen classes)
        logits = output['emo_from_au']
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities

    def _predict_fused(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        masks: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict using fused multimodal features

        Returns:
            predictions: [batch_size]
            probabilities: [batch_size, num_classes]
        """
        output = self.model(text_features, audio_features, video_features, masks)
        logits = output['emo_from_au']
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        return predictions, probabilities

    def _check_all_agree(
        self,
        predictions: Dict[str, torch.Tensor],
        probabilities: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        All 4 predictions must agree

        Returns:
            is_consistent: [batch_size]
            consensus_label: [batch_size]
            confidence: [batch_size]
        """
        batch_size = predictions['text'].shape[0]
        device = predictions['text'].device

        # Stack predictions
        pred_stack = torch.stack([
            predictions['text'],
            predictions['audio'],
            predictions['video'],
            predictions['fused']
        ])  # [4, batch_size]

        # Check if all agree
        is_consistent = (pred_stack == pred_stack[0]).all(dim=0)  # [batch_size]

        # Consensus is the agreed label (or first one if not consistent)
        consensus_label = predictions['fused']

        # Confidence is average of fused probability
        confidence = probabilities['fused'].max(dim=1)[0]

        # Filter by minimum confidence
        is_consistent = is_consistent & (confidence >= self.min_confidence)

        return is_consistent, consensus_label, confidence

    def _check_majority(
        self,
        predictions: Dict[str, torch.Tensor],
        probabilities: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        At least 3/4 predictions must agree

        Returns:
            is_consistent: [batch_size]
            consensus_label: [batch_size]
            confidence: [batch_size]
        """
        batch_size = predictions['text'].shape[0]
        device = predictions['text'].device
        num_classes = probabilities['fused'].shape[1]

        # Stack predictions
        pred_stack = torch.stack([
            predictions['text'],
            predictions['audio'],
            predictions['video'],
            predictions['fused']
        ])  # [4, batch_size]

        # Count votes for each class
        consensus_label = torch.zeros(batch_size, dtype=torch.long, device=device)
        vote_counts = torch.zeros(batch_size, dtype=torch.long, device=device)

        for i in range(batch_size):
            votes = pred_stack[:, i]
            unique, counts = torch.unique(votes, return_counts=True)
            max_count_idx = torch.argmax(counts)
            consensus_label[i] = unique[max_count_idx]
            vote_counts[i] = counts[max_count_idx]

        # At least 3/4 must agree
        is_consistent = vote_counts >= 3

        # Confidence is the fused prediction probability for consensus label
        confidence = probabilities['fused'][
            torch.arange(batch_size),
            consensus_label
        ]

        # Filter by minimum confidence
        is_consistent = is_consistent & (confidence >= self.min_confidence)

        return is_consistent, consensus_label, confidence

    def _check_weighted_vote(
        self,
        predictions: Dict[str, torch.Tensor],
        probabilities: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Weighted voting by prediction confidence

        Returns:
            is_consistent: [batch_size]
            consensus_label: [batch_size]
            confidence: [batch_size]
        """
        batch_size = predictions['text'].shape[0]
        device = predictions['text'].device
        num_classes = probabilities['fused'].shape[1]

        # Accumulate weighted votes
        vote_weights = torch.zeros(batch_size, num_classes, device=device)

        for modality in ['text', 'audio', 'video', 'fused']:
            pred = predictions[modality]
            prob = probabilities[modality]

            # Weight is the prediction confidence
            weight = prob.max(dim=1)[0].unsqueeze(1)  # [batch, 1]

            # Add weighted votes
            vote_weights.scatter_add_(
                1,
                pred.unsqueeze(1),
                weight
            )

        # Consensus is the class with highest weighted votes
        consensus_label = torch.argmax(vote_weights, dim=1)
        max_weight = vote_weights.max(dim=1)[0]
        total_weight = vote_weights.sum(dim=1)

        # Confidence is the ratio of max weight to total weight
        confidence = max_weight / (total_weight + 1e-10)

        # Consistent if confidence is high
        is_consistent = confidence >= self.min_confidence

        return is_consistent, consensus_label, confidence

    def _check_entropy(
        self,
        probabilities: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Check consistency based on prediction entropy

        Low entropy = high confidence = consistent

        Returns:
            is_consistent: [batch_size]
            consensus_label: [batch_size]
            confidence: [batch_size]
        """
        batch_size = probabilities['fused'].shape[0]

        # Use fused prediction
        prob = probabilities['fused']

        # Calculate entropy: H = -Σ p*log(p)
        entropy = -(prob * torch.log(prob + 1e-10)).sum(dim=1)  # [batch_size]

        # Low entropy = consistent
        is_consistent = entropy <= self.entropy_threshold

        # Consensus label
        consensus_label = torch.argmax(prob, dim=1)

        # Confidence is max probability
        confidence = prob.max(dim=1)[0]

        # Also check minimum confidence
        is_consistent = is_consistent & (confidence >= self.min_confidence)

        return is_consistent, consensus_label, confidence

    def _check_combined(
        self,
        predictions: Dict[str, torch.Tensor],
        probabilities: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Combined strategy: majority vote + entropy threshold

        Returns:
            is_consistent: [batch_size]
            consensus_label: [batch_size]
            confidence: [batch_size]
        """
        # Get results from both strategies
        is_maj, label_maj, conf_maj = self._check_majority(predictions, probabilities)
        is_ent, label_ent, conf_ent = self._check_entropy(probabilities)

        # Combine: both must agree
        is_consistent = is_maj & is_ent & (label_maj == label_ent)
        consensus_label = label_maj
        confidence = (conf_maj + conf_ent) / 2

        return is_consistent, consensus_label, confidence

    def get_statistics(self) -> Dict:
        """Get consistency checking statistics"""
        consistency_rate = (
            self.total_consistent / self.total_checked
            if self.total_checked > 0 else 0.0
        )

        return {
            'total_checked': self.total_checked,
            'total_consistent': self.total_consistent,
            'consistency_rate': consistency_rate,
            'strategy': self.strategy.value,
            'min_confidence': self.min_confidence
        }

    def reset_statistics(self):
        """Reset statistics counters"""
        self.total_checked = 0
        self.total_consistent = 0
        self.consistency_by_class = {}


class AdaptiveConsistencyChecker(MultimodalConsistencyChecker):
    """
    Adaptive Consistency Checker

    Automatically adjusts consistency thresholds based on
    observed performance and distribution of predictions.

    Parameters:
    -----------
    model : nn.Module
        The main network
    initial_strategy : ConsistencyStrategy
        Initial consistency strategy
    initial_confidence : float
        Initial minimum confidence threshold
    adaptation_rate : float
        Rate of threshold adaptation (0-1)
    target_consistency_rate : float
        Target consistency rate to maintain
    """

    def __init__(
        self,
        model: nn.Module,
        initial_strategy: ConsistencyStrategy = ConsistencyStrategy.MAJORITY,
        initial_confidence: float = 0.7,
        adaptation_rate: float = 0.1,
        target_consistency_rate: float = 0.3
    ):
        super().__init__(model, initial_strategy, initial_confidence)

        self.adaptation_rate = adaptation_rate
        self.target_consistency_rate = target_consistency_rate
        self.confidence_history = []

    def adapt_threshold(self):
        """
        Adapt confidence threshold based on observed consistency rate

        If consistency rate is too low, lower the threshold
        If consistency rate is too high, raise the threshold
        """
        if self.total_checked < 100:  # Need enough samples
            return

        current_rate = self.total_consistent / self.total_checked

        # Adjust threshold
        if current_rate < self.target_consistency_rate:
            # Too strict, lower threshold
            self.min_confidence *= (1 - self.adaptation_rate)
        elif current_rate > self.target_consistency_rate * 1.5:
            # Too lenient, raise threshold
            self.min_confidence *= (1 + self.adaptation_rate)

        # Clamp to reasonable range
        self.min_confidence = max(0.5, min(0.95, self.min_confidence))

        # Record history
        self.confidence_history.append({
            'step': self.total_checked,
            'threshold': self.min_confidence,
            'consistency_rate': current_rate
        })

    def check_consistency(self, *args, **kwargs):
        """Override to include adaptation"""
        result = super().check_consistency(*args, **kwargs)

        # Adapt threshold every 100 samples
        if self.total_checked % 100 == 0:
            self.adapt_threshold()

        return result


if __name__ == "__main__":
    print("Testing Multimodal Consistency Checker...")

    # This would require a full model, so we'll just test the logic
    print("Consistency strategies:")
    for strategy in ConsistencyStrategy:
        print(f"  - {strategy.value}")

    print("\n✓ Consistency checker module ready!")
