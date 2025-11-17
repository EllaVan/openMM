"""
Multimodal Domain-Specific Prompts for Continual Learning

Inspired by S-Prompts (NeurIPS 2022), adapted for multimodal emotion recognition.

Key Ideas:
1. Learn domain-specific prompts for each modality (text, audio, video)
2. Store prompts in a growing pool (one set per domain)
3. At test time, retrieve prompts using K-NN on AU prototypes
4. Prepend retrieved prompts to input features

References:
- S-Prompts: Wang et al., NeurIPS 2022
- "S-Prompts Learning with Pre-trained Transformers:
   An Occam's Razor for Domain Incremental Learning"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors


class ModalityPrompt(nn.Module):
    """
    Domain-specific prompt for a single modality

    Prompts are prepended to the input sequence:
    [prompt_1, prompt_2, ..., prompt_L, x_1, x_2, ..., x_T]

    This allows the model to adapt its processing based on domain
    without modifying the original features.
    """

    def __init__(
        self,
        prompt_length: int = 5,
        feature_dim: int = 768,
        device: str = 'cuda'
    ):
        """
        Args:
            prompt_length: Number of prompt tokens
            feature_dim: Dimension of each prompt token
            device: Device for tensors
        """
        super().__init__()

        self.prompt_length = prompt_length
        self.feature_dim = feature_dim

        # Initialize prompts with small random values
        self.prompts = nn.Parameter(
            torch.randn(prompt_length, feature_dim, device=device) * 0.01
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepend prompts to input sequence

        Args:
            x: [batch, seq_len, feature_dim]

        Returns:
            x_with_prompts: [batch, prompt_length + seq_len, feature_dim]
        """
        batch_size = x.shape[0]

        # Expand prompts for batch
        prompts_expanded = self.prompts.unsqueeze(0).expand(
            batch_size, -1, -1
        )  # [batch, prompt_length, feature_dim]

        # Prepend to input
        x_with_prompts = torch.cat([prompts_expanded, x], dim=1)

        return x_with_prompts


class MultimodalDomainPrompts(nn.Module):
    """
    Domain-specific prompts for all modalities

    Manages prompts for text, audio, and video modalities,
    storing one set per domain.
    """

    def __init__(
        self,
        text_prompt_length: int = 5,
        audio_prompt_length: int = 5,
        video_prompt_length: int = 5,
        text_dim: int = 768,
        audio_dim: int = 768,
        video_dim: int = 768,
        device: str = 'cuda'
    ):
        super().__init__()

        self.text_prompt_length = text_prompt_length
        self.audio_prompt_length = audio_prompt_length
        self.video_prompt_length = video_prompt_length

        # Store prompts for each domain
        # domain_id -> {text_prompt, audio_prompt, video_prompt}
        self.domain_prompts = nn.ModuleDict()

        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.device_str = device

    def add_domain(self, domain_id: str):
        """Add prompts for a new domain"""
        if domain_id not in self.domain_prompts:
            domain_prompts = nn.ModuleDict({
                'text': ModalityPrompt(
                    self.text_prompt_length, self.text_dim, self.device_str
                ),
                'audio': ModalityPrompt(
                    self.audio_prompt_length, self.audio_dim, self.device_str
                ),
                'video': ModalityPrompt(
                    self.video_prompt_length, self.video_dim, self.device_str
                )
            })

            self.domain_prompts[domain_id] = domain_prompts
            print(f"Added domain prompts for: {domain_id}")

    def get_prompts(self, domain_id: str) -> Dict[str, ModalityPrompt]:
        """Get prompts for a specific domain"""
        if domain_id not in self.domain_prompts:
            raise ValueError(f"Domain {domain_id} not found")
        return self.domain_prompts[domain_id]

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        domain_id: str
    ) -> Dict[str, torch.Tensor]:
        """
        Apply domain-specific prompts to all modalities

        Args:
            text: [batch, seq_len, text_dim]
            audio: [batch, seq_len, audio_dim]
            video: [batch, seq_len, video_dim]
            domain_id: Which domain prompts to use

        Returns:
            Dict with prompted features for each modality
        """
        prompts = self.get_prompts(domain_id)

        text_prompted = prompts['text'](text)
        audio_prompted = prompts['audio'](audio)
        video_prompted = prompts['video'](video)

        return {
            'text': text_prompted,
            'audio': audio_prompted,
            'video': video_prompted
        }


class AUPrototypeBank:
    """
    AU Prototype Bank for Domain Retrieval

    Inspired by S-Prompts' K-NN retrieval, but uses AU activations
    as the feature space instead of raw image features.

    Why AU prototypes?
    - AUs are more stable across domains than raw features
    - AU space has semantic meaning (facial expressions)
    - Enables zero-shot domain detection
    """

    def __init__(
        self,
        num_prototypes_per_domain: int = 10,
        num_aus: int = 23,
        k_neighbors: int = 3,
        device: str = 'cuda'
    ):
        """
        Args:
            num_prototypes_per_domain: Number of prototypes per domain
            num_aus: Number of Action Units
            k_neighbors: K for K-NN retrieval
            device: Device for tensors
        """
        self.num_prototypes_per_domain = num_prototypes_per_domain
        self.num_aus = num_aus
        self.k_neighbors = k_neighbors
        self.device = device

        # domain_id -> prototypes [num_prototypes, num_aus]
        self.prototypes = {}

        # domain_id -> domain_center [num_aus]
        self.domain_centers = {}

        # For K-NN retrieval
        self.knn_model = None
        self.domain_labels = []  # Maps prototype index to domain_id

    def build_prototypes(
        self,
        domain_id: str,
        au_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ):
        """
        Build prototypes for a domain using K-Means clustering

        Args:
            domain_id: Domain identifier
            au_features: [num_samples, num_aus] AU activations
            labels: Optional emotion labels for class-balanced sampling
        """
        au_features_np = au_features.detach().cpu().numpy()

        # Use K-Means to find prototypes
        kmeans = KMeans(
            n_clusters=self.num_prototypes_per_domain,
            random_state=42,
            n_init=10
        )
        kmeans.fit(au_features_np)

        # Store cluster centers as prototypes
        prototypes = torch.tensor(
            kmeans.cluster_centers_,
            dtype=torch.float32,
            device=self.device
        )

        self.prototypes[domain_id] = prototypes

        # Store domain center (mean of all prototypes)
        domain_center = prototypes.mean(dim=0)
        self.domain_centers[domain_id] = domain_center

        print(f"Built {self.num_prototypes_per_domain} prototypes for domain: {domain_id}")

        # Rebuild K-NN model
        self._rebuild_knn()

    def _rebuild_knn(self):
        """Rebuild K-NN model with all current prototypes"""
        if len(self.prototypes) == 0:
            return

        # Concatenate all prototypes
        all_prototypes = []
        self.domain_labels = []

        for domain_id, prototypes in self.prototypes.items():
            all_prototypes.append(prototypes.cpu().numpy())
            self.domain_labels.extend([domain_id] * len(prototypes))

        all_prototypes = np.vstack(all_prototypes)

        # Build K-NN model
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.k_neighbors, len(all_prototypes)),
            metric='cosine'
        )
        self.knn_model.fit(all_prototypes)

        print(f"Rebuilt K-NN model with {len(all_prototypes)} prototypes from {len(self.prototypes)} domains")

    def retrieve_domain(
        self,
        au_features: torch.Tensor,
        return_distances: bool = False
    ) -> str:
        """
        Retrieve most similar domain for given AU features

        Args:
            au_features: [num_aus] or [batch, num_aus]
            return_distances: Whether to return distances

        Returns:
            domain_id: Most similar domain
            distances: (optional) Distances to nearest prototypes
        """
        if self.knn_model is None:
            raise ValueError("No prototypes built yet. Call build_prototypes() first.")

        # Handle batch input
        if au_features.dim() == 2:
            # Use mean AU features for the batch
            au_features = au_features.mean(dim=0)

        au_features_np = au_features.detach().cpu().numpy().reshape(1, -1)

        # Find K nearest prototypes
        distances, indices = self.knn_model.kneighbors(au_features_np)

        # Vote: which domain appears most in K neighbors?
        neighbor_domains = [self.domain_labels[idx] for idx in indices[0]]
        domain_id = max(set(neighbor_domains), key=neighbor_domains.count)

        if return_distances:
            return domain_id, distances[0]
        else:
            return domain_id

    def get_domain_center_similarity(
        self,
        au_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute similarity to each domain center

        Args:
            au_features: [num_aus] or [batch, num_aus]

        Returns:
            Dict mapping domain_id to similarity score
        """
        if au_features.dim() == 2:
            au_features = au_features.mean(dim=0)

        similarities = {}

        for domain_id, center in self.domain_centers.items():
            # Cosine similarity
            sim = F.cosine_similarity(
                au_features.unsqueeze(0),
                center.unsqueeze(0),
                dim=1
            ).item()
            similarities[domain_id] = sim

        return similarities


def test_multimodal_prompts():
    """Test multimodal domain prompts"""
    print("Testing Multimodal Domain Prompts...")

    # Create prompt manager
    prompts = MultimodalDomainPrompts(
        text_prompt_length=5,
        audio_prompt_length=5,
        video_prompt_length=5,
        device='cpu'
    )

    # Add domains
    prompts.add_domain('domain_0')
    prompts.add_domain('domain_1')

    # Test forward pass
    batch_size = 4
    seq_len = 50

    text = torch.randn(batch_size, seq_len, 768)
    audio = torch.randn(batch_size, seq_len, 768)
    video = torch.randn(batch_size, seq_len, 768)

    # Apply prompts for domain_0
    prompted = prompts(text, audio, video, 'domain_0')

    print(f"\nOriginal text shape: {text.shape}")
    print(f"Prompted text shape: {prompted['text'].shape}")
    print(f"Expected: {(batch_size, 5 + seq_len, 768)}")

    assert prompted['text'].shape == (batch_size, 5 + seq_len, 768)
    print("✓ Prompt prepending works correctly")

    # Test AU prototype bank
    print("\nTesting AU Prototype Bank...")

    bank = AUPrototypeBank(num_prototypes_per_domain=10, num_aus=23, device='cpu')

    # Build prototypes for domain_0
    au_features_d0 = torch.randn(100, 23)
    bank.build_prototypes('domain_0', au_features_d0)

    # Build prototypes for domain_1
    au_features_d1 = torch.randn(100, 23) + 2.0  # Shifted distribution
    bank.build_prototypes('domain_1', au_features_d1)

    # Test retrieval
    test_au = au_features_d1[0]  # Should retrieve domain_1
    retrieved_domain = bank.retrieve_domain(test_au)
    print(f"Test AU retrieved domain: {retrieved_domain}")

    # Get similarities
    similarities = bank.get_domain_center_similarity(test_au)
    print(f"Domain similarities: {similarities}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_multimodal_prompts()
