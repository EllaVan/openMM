"""
Multi-Head Multimodal Network for Domain Incremental Learning

Inspired by DILLB (Domain Incremental Learning for object detection),
this module implements a multi-head architecture for multimodal emotion recognition.

Key Features:
1. Shared multimodal encoder (backbone) - domain-invariant features
2. Shared AU predictor - universal emotion-related representation
3. Multiple emotion classification heads - domain-specific classifiers
4. Global + domain-specific AU-EMO matrices

Architecture:
    Input (Text, Audio, Video)
         ↓
    Shared Multimodal Encoder (frozen after Task 0, optional)
         ↓
    Shared AU Predictor (always trainable)
         ↓
    ├─ Global AU-EMO Matrix (shared across all domains)
    └─ Domain-specific AU-EMO Matrices (one per task)
         ↓
    Multiple Emotion Heads (one per domain/task)

References:
- DILLB: https://github.com/Disguiser15/DILLB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from continual_learning.au_emotion_network import AUPredictor
from hyper_fusion.hypergraph_model import HypergraphMultimodalFusion


class MultiHeadAUEMOMatrix(nn.Module):
    """
    Multi-head AU-EMO matrix manager

    Maintains:
    1. Global shared matrix (across all domains)
    2. Domain-specific matrices (one per task)

    Prediction uses weighted combination:
        P(EMO|AU) = α * P_global(EMO|AU) + (1-α) * P_domain(EMO|AU)
    """

    def __init__(
        self,
        num_aus: int = 23,
        num_emotions: int = 6,
        prior_p_au_given_emo: Optional[torch.Tensor] = None,
        global_weight: float = 0.5,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.global_weight = global_weight
        self.device_str = device

        # Initialize prior
        if prior_p_au_given_emo is None:
            prior_p_au_emo = torch.ones(num_aus, num_emotions) / num_emotions
        else:
            prior_p_au_emo = prior_p_au_given_emo

        # Convert P(AU|EMO) to P(EMO|AU)
        prior_p_emo_au = prior_p_au_emo / (prior_p_au_emo.sum(dim=1, keepdim=True) + 1e-10)

        # Global shared matrix (learnable)
        self.global_matrix = nn.Parameter(
            torch.log(prior_p_emo_au + 1e-10).to(device)
        )

        # Domain-specific matrices (stored in ModuleDict)
        self.domain_matrices = nn.ModuleDict()

        # Store prior
        self.register_buffer('prior_probs', prior_p_emo_au.to(device))

    def add_domain(self, domain_id: str):
        """Add a new domain-specific matrix"""
        if domain_id not in self.domain_matrices:
            # Initialize with prior
            domain_matrix = nn.Parameter(
                torch.log(self.prior_probs + 1e-10).clone()
            )
            self.domain_matrices[domain_id] = nn.ParameterList([domain_matrix])
            print(f"Added domain-specific matrix for domain: {domain_id}")

    def get_matrix(self, domain_id: Optional[str] = None) -> torch.Tensor:
        """
        Get probability matrix for prediction

        If domain_id is None: use global matrix only
        If domain_id provided: weighted combination of global + domain-specific
        """
        # Global matrix
        p_global = F.softmax(self.global_matrix, dim=1)

        if domain_id is None or domain_id not in self.domain_matrices:
            return p_global

        # Domain-specific matrix
        domain_logits = self.domain_matrices[domain_id][0]
        p_domain = F.softmax(domain_logits, dim=1)

        # Weighted combination
        p_combined = self.global_weight * p_global + (1 - self.global_weight) * p_domain

        return p_combined

    def forward(self, au_probs: torch.Tensor, domain_id: Optional[str] = None) -> torch.Tensor:
        """
        Predict emotions from AU probabilities

        Args:
            au_probs: [batch, num_aus]
            domain_id: which domain to use (None for global only)

        Returns:
            emo_logits: [batch, num_emotions]
        """
        p_emo_given_au = self.get_matrix(domain_id)  # [num_aus, num_emotions]
        emo_logits = torch.matmul(au_probs, p_emo_given_au)  # [batch, num_emotions]
        return emo_logits


class MultiHeadEmotionClassifier(nn.Module):
    """
    Multi-head emotion classifier

    One classification head per domain/task, enabling:
    - Domain-specific decision boundaries
    - No interference between domains
    - Easy addition of new domains
    """

    def __init__(
        self,
        input_dim: int = 768,
        num_emotions: int = 6,
        hidden_dim: int = 256,
        device: str = 'cuda'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_emotions = num_emotions
        self.hidden_dim = hidden_dim
        self.device_str = device

        # Domain-specific heads (stored in ModuleDict)
        self.domain_heads = nn.ModuleDict()

    def add_domain(self, domain_id: str):
        """Add a new domain-specific classification head"""
        if domain_id not in self.domain_heads:
            head = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(self.hidden_dim, self.num_emotions)
            ).to(self.device_str)

            self.domain_heads[domain_id] = head
            print(f"Added emotion classification head for domain: {domain_id}")

    def forward(self, features: torch.Tensor, domain_id: str) -> torch.Tensor:
        """
        Classify emotions using domain-specific head

        Args:
            features: [batch, input_dim]
            domain_id: which head to use

        Returns:
            logits: [batch, num_emotions]
        """
        if domain_id not in self.domain_heads:
            raise ValueError(f"Domain {domain_id} not registered. Call add_domain() first.")

        return self.domain_heads[domain_id](features)


class MultiHeadMultimodalNetwork(nn.Module):
    """
    Complete multi-head multimodal network for domain incremental learning

    DILLB-inspired architecture:
    1. Shared backbone (multimodal encoder) - optionally frozen
    2. Shared AU predictor - always trainable
    3. Multi-head AU-EMO matrices (global + domain-specific)
    4. Multi-head emotion classifiers (one per domain)

    Training modes:
    - Task 0: Train everything from scratch
    - Task 1+: Optionally freeze backbone, train AU predictor + new heads
    """

    def __init__(
        self,
        text_input_dim: int = 768,
        audio_input_dim: int = 768,
        video_input_dim: int = 768,
        num_aus: int = 23,
        num_emotions: int = 6,
        encoder_hidden_dim: int = 256,
        num_hyperedges: int = 64,
        num_conv_layers: int = 2,
        au_emo_prior: Optional[torch.Tensor] = None,
        freeze_backbone: bool = False,
        global_matrix_weight: float = 0.5,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.freeze_backbone = freeze_backbone
        self.device_str = device

        # 1. Shared Multimodal Encoder (Backbone)
        self.multimodal_encoder = HypergraphMultimodalFusion(
            text_input_dim=text_input_dim,
            audio_input_dim=audio_input_dim,
            video_input_dim=video_input_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            num_hyperedges=num_hyperedges,
            num_conv_layers=num_conv_layers,
            use_bottleneck=True,
            device=device
        )

        # Optionally freeze backbone after Task 0
        if freeze_backbone:
            self._freeze_backbone()

        # 2. Shared AU Predictor (always trainable)
        self.au_predictor = AUPredictor(
            input_dim=768,  # After bottleneck
            num_aus=num_aus,
            hidden_dim=256,
            device=device
        )

        # 3. Multi-head AU-EMO Matrices
        self.multi_head_au_emo = MultiHeadAUEMOMatrix(
            num_aus=num_aus,
            num_emotions=num_emotions,
            prior_p_au_given_emo=au_emo_prior,
            global_weight=global_matrix_weight,
            device=device
        )

        # 4. Multi-head Emotion Classifiers
        self.multi_head_classifier = MultiHeadEmotionClassifier(
            input_dim=768,
            num_emotions=num_emotions,
            hidden_dim=256,
            device=device
        )

        # Track registered domains
        self.registered_domains = set()

    def add_domain(self, domain_id: str):
        """Register a new domain (add matrices and heads)"""
        if domain_id not in self.registered_domains:
            self.multi_head_au_emo.add_domain(domain_id)
            self.multi_head_classifier.add_domain(domain_id)
            self.registered_domains.add(domain_id)
            print(f"Domain {domain_id} registered successfully")

    def _freeze_backbone(self):
        """Freeze multimodal encoder to preserve learned features"""
        for param in self.multimodal_encoder.parameters():
            param.requires_grad = False
        print("Multimodal encoder (backbone) frozen")

    def _unfreeze_backbone(self):
        """Unfreeze multimodal encoder for training"""
        for param in self.multimodal_encoder.parameters():
            param.requires_grad = True
        print("Multimodal encoder (backbone) unfrozen")

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        masks: Optional[Dict] = None,
        domain_id: Optional[str] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            text: [batch, seq_len, text_dim]
            audio: [batch, seq_len, audio_dim]
            video: [batch, seq_len, video_dim]
            masks: optional masks for sequences
            domain_id: which domain head to use (required for emotion prediction)
            return_features: whether to return intermediate features

        Returns:
            dict with:
                - au_probs: [batch, num_aus]
                - emo_from_au: [batch, num_emotions] (via AU-EMO matrix)
                - emo_direct: [batch, num_emotions] (via domain-specific head)
                - multimodal_features: [batch, 768] (if return_features=True)
        """
        # 1. Multimodal encoding (shared backbone)
        multimodal_features = self.multimodal_encoder(text, audio, video, masks)
        # Output: [batch, 768]

        # 2. AU prediction (shared predictor)
        au_probs = self.au_predictor(multimodal_features)
        # Output: [batch, 23]

        # 3. Emotion via AU-EMO matrix
        emo_from_au = self.multi_head_au_emo(au_probs, domain_id)
        # Output: [batch, 6]

        # 4. Direct emotion prediction (domain-specific head)
        if domain_id is not None:
            emo_direct = self.multi_head_classifier(multimodal_features, domain_id)
        else:
            # If no domain specified, use global matrix only
            emo_direct = None

        outputs = {
            'au_probs': au_probs,
            'emo_from_au': emo_from_au,
            'emo_direct': emo_direct
        }

        if return_features:
            outputs['multimodal_features'] = multimodal_features

        return outputs

    def get_domain_parameters(self, domain_id: str) -> List[nn.Parameter]:
        """Get parameters specific to a domain (for selective optimization)"""
        params = []

        # Domain-specific AU-EMO matrix
        if domain_id in self.multi_head_au_emo.domain_matrices:
            params.extend(self.multi_head_au_emo.domain_matrices[domain_id].parameters())

        # Domain-specific emotion head
        if domain_id in self.multi_head_classifier.domain_heads:
            params.extend(self.multi_head_classifier.domain_heads[domain_id].parameters())

        return params

    def get_shared_parameters(self) -> List[nn.Parameter]:
        """Get shared parameters (AU predictor + global matrix + optionally backbone)"""
        params = []

        # AU predictor (always shared)
        params.extend(self.au_predictor.parameters())

        # Global AU-EMO matrix
        params.append(self.multi_head_au_emo.global_matrix)

        # Backbone (if not frozen)
        if not self.freeze_backbone:
            params.extend(self.multimodal_encoder.parameters())

        return params


def test_multi_head_network():
    """Test multi-head network"""
    print("Testing Multi-Head Multimodal Network...")

    # Create network
    network = MultiHeadMultimodalNetwork(
        num_aus=23,
        num_emotions=6,
        freeze_backbone=False,
        device='cpu'
    )

    print(f"\nInitial registered domains: {network.registered_domains}")

    # Add domains
    network.add_domain('task_0')
    network.add_domain('task_1')

    print(f"After adding domains: {network.registered_domains}")

    # Test forward pass
    batch_size = 4
    seq_len = 50
    text = torch.randn(batch_size, seq_len, 768)
    audio = torch.randn(batch_size, seq_len, 768)
    video = torch.randn(batch_size, seq_len, 768)

    # Task 0 prediction
    outputs_0 = network(text, audio, video, domain_id='task_0')
    print(f"\nTask 0 outputs:")
    print(f"  AU probs shape: {outputs_0['au_probs'].shape}")
    print(f"  EMO from AU shape: {outputs_0['emo_from_au'].shape}")
    print(f"  EMO direct shape: {outputs_0['emo_direct'].shape}")

    # Task 1 prediction
    outputs_1 = network(text, audio, video, domain_id='task_1')
    print(f"\nTask 1 outputs:")
    print(f"  AU probs shape: {outputs_1['au_probs'].shape}")
    print(f"  EMO from AU shape: {outputs_1['emo_from_au'].shape}")
    print(f"  EMO direct shape: {outputs_1['emo_direct'].shape}")

    # Check parameter counts
    total_params = sum(p.numel() for p in network.parameters())
    shared_params = sum(p.numel() for p in network.get_shared_parameters())
    domain_0_params = sum(p.numel() for p in network.get_domain_parameters('task_0'))

    print(f"\nParameter counts:")
    print(f"  Total: {total_params:,}")
    print(f"  Shared: {shared_params:,}")
    print(f"  Domain-specific (task_0): {domain_0_params:,}")

    print("\n✓ Multi-head network test passed!")


if __name__ == "__main__":
    test_multi_head_network()
