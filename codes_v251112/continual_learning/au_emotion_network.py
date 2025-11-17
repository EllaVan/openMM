"""
AU-based Emotion Recognition Network for Continual Learning

Extends the MultimodalHypergraphFusion with AU prediction branch
and AU-EMO matrix for zero-shot continual learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from hyper_fusion.network import (
    UnimodalEncoder,
    MultimodalHypergraphFusion
)
from hyper_fusion.modules import (
    MultimodalHypergraphLayer,
    GraphContrastiveLearning,
    BottleneckLayer
)
from continual_learning.au_emo_matrix import AUEMOMatrix


class AUPredictor(nn.Module):
    """
    Action Unit Predictor based on multimodal features

    Predicts AU activation probabilities from fused multimodal features.
    Uses multi-label classification (sigmoid) since multiple AUs can
    be active simultaneously.

    Parameters:
    -----------
    input_dim : int
        Input feature dimension (from pooled multimodal features)
    num_aus : int
        Number of Action Units to predict
    hidden_dim : int
        Hidden layer dimension
    dropout : float
        Dropout rate
    """

    def __init__(
        self,
        input_dim: int,
        num_aus: int = 23,
        hidden_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_aus = num_aus

        # Multi-layer predictor
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_aus)
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict AU probabilities

        Args:
            features: [batch_size, input_dim]

        Returns:
            au_probs: [batch_size, num_aus]
                Sigmoid probabilities for each AU
        """
        logits = self.predictor(features)
        au_probs = torch.sigmoid(logits)  # Multi-label classification
        return au_probs


class AUEmotionNetwork(nn.Module):
    """
    Complete AU-based Emotion Recognition Network

    Architecture:
    1. Multimodal feature extraction (text, audio, video)
    2. Multimodal hypergraph fusion
    3. AU prediction branch
    4. Emotion prediction via AU-EMO matrix
    5. Direct emotion prediction (for comparison)

    This network supports both seen and unseen emotion classes
    through the AU-EMO probability matrix.

    Parameters:
    -----------
    text_input_dim : int
        Text feature dimension
    audio_input_dim : int
        Audio feature dimension
    video_input_dim : int
        Video feature dimension
    num_aus : int
        Number of Action Units
    num_emotions : int
        Number of emotion classes (excluding neutral)
    au_emo_prior : torch.Tensor or np.ndarray, optional
        Prior AU-EMO probability matrix [num_aus, num_emotions]
    encoder_hidden_dim : int
        Encoder hidden dimension
    encoder_output_dim : int
        Encoder output dimension
    hypergraph_hidden_dim : int
        Hypergraph hidden dimension
    num_hyperedges : int
        Number of hyperedges
    num_conv_layers : int
        Number of hypergraph convolution layers
    bottleneck_dim : int
        Bottleneck dimension
    dropout : float
        Dropout rate
    hyperedge_drop_rate : float
        Hyperedge dropout rate
    use_contrastive : bool
        Whether to use contrastive learning
    use_bottleneck : bool
        Whether to use bottleneck layer
    contrastive_weight : float
        Weight for contrastive loss
    au_emo_prior_strength : float
        Strength of AU-EMO prior
    """

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        video_input_dim: int,
        num_aus: int = 23,
        num_emotions: int = 6,
        au_emo_prior: Optional[torch.Tensor] = None,
        encoder_hidden_dim: int = 256,
        encoder_output_dim: int = 256,
        hypergraph_hidden_dim: int = 256,
        num_hyperedges: int = 64,
        num_conv_layers: int = 2,
        bottleneck_dim: int = 128,
        dropout: float = 0.1,
        hyperedge_drop_rate: float = 0.2,
        use_contrastive: bool = True,
        use_bottleneck: bool = True,
        contrastive_weight: float = 0.1,
        au_emo_prior_strength: float = 100.0,
        device: str = 'cuda'
    ):
        super().__init__()

        self.num_aus = num_aus
        self.num_emotions = num_emotions
        self.use_contrastive = use_contrastive
        self.use_bottleneck = use_bottleneck
        self.contrastive_weight = contrastive_weight
        self.device_str = device

        # 1. Single-modal encoders
        self.text_encoder = UnimodalEncoder(
            input_dim=text_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        self.audio_encoder = UnimodalEncoder(
            input_dim=audio_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        self.video_encoder = UnimodalEncoder(
            input_dim=video_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        # 2. Multimodal hypergraph layer
        self.hypergraph = MultimodalHypergraphLayer(
            text_dim=encoder_output_dim,
            audio_dim=encoder_output_dim,
            video_dim=encoder_output_dim,
            hidden_dim=hypergraph_hidden_dim,
            num_hyperedges=num_hyperedges,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            hyperedge_drop_rate=hyperedge_drop_rate
        )

        # 3. Bottleneck layer (optional)
        if use_bottleneck:
            self.bottleneck = BottleneckLayer(
                in_dim=hypergraph_hidden_dim,
                bottleneck_dim=bottleneck_dim,
                out_dim=hypergraph_hidden_dim
            )

        # 4. Graph contrastive learning (optional)
        if use_contrastive:
            self.contrastive = GraphContrastiveLearning(
                feature_dim=hypergraph_hidden_dim,
                projection_dim=128
            )

        # 5. AU predictor (NEW)
        multimodal_feature_dim = hypergraph_hidden_dim * 3
        self.au_predictor = AUPredictor(
            input_dim=multimodal_feature_dim,
            num_aus=num_aus,
            hidden_dim=256,
            dropout=dropout
        )

        # 6. AU-EMO probability matrix (NEW)
        self.au_emo_matrix = AUEMOMatrix(
            num_aus=num_aus,
            num_emotions=num_emotions,
            prior_matrix=au_emo_prior,
            prior_strength=au_emo_prior_strength,
            device=device
        )

        # 7. Direct emotion classifier (for comparison)
        self.direct_classifier = nn.Sequential(
            nn.Linear(multimodal_feature_dim, hypergraph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hypergraph_hidden_dim, num_emotions)
        )

        # L2 regularization weight
        self.l2_reg_weight = 0.001

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_H: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            text_features: [batch_size, T, text_input_dim]
            audio_features: [batch_size, T, audio_input_dim]
            video_features: [batch_size, T, video_input_dim]
            masks: [batch_size, T] - Valid frame mask
            labels: [batch_size] - Emotion labels (optional)
            return_H: Whether to return hypergraph matrix

        Returns:
            dict: {
                'au_probs': AU probabilities [batch_size, num_aus],
                'emo_from_au': Emotion prediction from AU path [batch_size, num_emotions],
                'emo_direct': Direct emotion prediction [batch_size, num_emotions],
                'au_emo_matrix': Current AU-EMO matrix [num_aus, num_emotions],
                'multimodal_feature': Pooled multimodal features [batch_size, hidden_dim*3],
                'text_pooled': Pooled text features [batch_size, hidden_dim],
                'audio_pooled': Pooled audio features [batch_size, hidden_dim],
                'video_pooled': Pooled video features [batch_size, hidden_dim],
                'H': Hypergraph matrix (if return_H=True),
                'loss': Total loss (if labels provided),
                'cls_loss': Classification loss,
                'contrastive_loss': Contrastive loss
            }
        """
        # 1. Single-modal encoding
        text_encoded = self.text_encoder(text_features, mask=masks)
        audio_encoded = self.audio_encoder(audio_features, mask=masks)
        video_encoded = self.video_encoder(video_features, mask=masks)

        # 2. Hypergraph fusion
        hypergraph_out = self.hypergraph(
            text_encoded,
            audio_encoded,
            video_encoded,
            mask=masks,
            return_H=return_H
        )

        fused_features = hypergraph_out['fused']  # [batch, 3T, hidden_dim]

        # 3. Bottleneck (optional)
        if self.use_bottleneck:
            fused_features, bottleneck_features = self.bottleneck(fused_features)

        # 4. Separate modalities and pool
        batch_size, total_nodes, hidden_dim = fused_features.shape
        T = total_nodes // 3

        text_nodes = fused_features[:, :T, :]
        audio_nodes = fused_features[:, T:2*T, :]
        video_nodes = fused_features[:, 2*T:, :]

        # Masked pooling
        if masks is not None:
            mask_expanded = masks.unsqueeze(-1).float()
            valid_counts = masks.sum(dim=1, keepdim=True).float()

            text_pooled = (text_nodes * mask_expanded).sum(dim=1) / valid_counts
            audio_pooled = (audio_nodes * mask_expanded).sum(dim=1) / valid_counts
            video_pooled = (video_nodes * mask_expanded).sum(dim=1) / valid_counts
        else:
            text_pooled = text_nodes.mean(dim=1)
            audio_pooled = audio_nodes.mean(dim=1)
            video_pooled = video_nodes.mean(dim=1)

        # 5. Concatenate multimodal features
        multimodal_feature = torch.cat(
            [text_pooled, audio_pooled, video_pooled],
            dim=1
        )  # [batch, hidden_dim * 3]

        # 6. AU prediction (NEW)
        au_probs = self.au_predictor(multimodal_feature)  # [batch, num_aus]

        # 7. Emotion prediction from AU (NEW)
        emo_from_au = self.au_emo_matrix(au_probs)  # [batch, num_emotions]

        # 8. Direct emotion prediction (for comparison)
        emo_direct = self.direct_classifier(multimodal_feature)  # [batch, num_emotions]

        # Prepare output
        output = {
            'au_probs': au_probs,
            'emo_from_au': emo_from_au,
            'emo_direct': emo_direct,
            'au_emo_matrix': self.au_emo_matrix.get_probability(),
            'multimodal_feature': multimodal_feature,
            'text_pooled': text_pooled,
            'audio_pooled': audio_pooled,
            'video_pooled': video_pooled
        }

        if return_H:
            output['H'] = hypergraph_out['H']

        # 9. Compute losses (if labels provided)
        if labels is not None:
            # Classification loss (both paths)
            cls_loss_au = F.cross_entropy(emo_from_au, labels)
            cls_loss_direct = F.cross_entropy(emo_direct, labels)
            cls_loss = cls_loss_au + cls_loss_direct

            output['cls_loss'] = cls_loss
            output['cls_loss_au'] = cls_loss_au
            output['cls_loss_direct'] = cls_loss_direct

            # Contrastive learning loss
            if self.use_contrastive:
                contrastive_loss = self.contrastive(multimodal_feature, labels)
                output['contrastive_loss'] = contrastive_loss
            else:
                contrastive_loss = torch.tensor(0.0, device=au_probs.device)
                output['contrastive_loss'] = contrastive_loss

            # L2 regularization
            l2_reg = sum(p.pow(2).sum() for p in self.parameters())
            output['l2_reg'] = l2_reg

            # Total loss
            total_loss = (
                cls_loss +
                self.contrastive_weight * contrastive_loss +
                self.l2_reg_weight * l2_reg
            )
            output['loss'] = total_loss

        return output

    def predict_from_au(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotions using AU path (for unseen classes)

        Returns:
            predictions: [batch_size] - Predicted emotion labels
            confidence: [batch_size] - Prediction confidence (max probability)
        """
        with torch.no_grad():
            output = self.forward(text_features, audio_features, video_features, masks)
            emo_probs = F.softmax(output['emo_from_au'], dim=1)
            predictions = torch.argmax(emo_probs, dim=1)
            confidence = torch.max(emo_probs, dim=1)[0]

        return predictions, confidence

    def predict_direct(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotions using direct path (for comparison)

        Returns:
            predictions: [batch_size]
            confidence: [batch_size]
        """
        with torch.no_grad():
            output = self.forward(text_features, audio_features, video_features, masks)
            emo_probs = F.softmax(output['emo_direct'], dim=1)
            predictions = torch.argmax(emo_probs, dim=1)
            confidence = torch.max(emo_probs, dim=1)[0]

        return predictions, confidence

    def update_au_emo_matrix(
        self,
        au_probs: torch.Tensor,
        emo_labels: torch.Tensor,
        is_seen: bool,
        confidence: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        Update AU-EMO matrix

        Wrapper for AUEMOMatrix.update()
        """
        return self.au_emo_matrix.update(
            au_probs, emo_labels, is_seen, confidence, **kwargs
        )

    def get_au_emo_statistics(self) -> Dict:
        """Get AU-EMO matrix statistics"""
        return self.au_emo_matrix.get_statistics()

    def save_au_emo_matrix(self, filepath: str):
        """Save AU-EMO matrix"""
        self.au_emo_matrix.save(filepath)

    def load_au_emo_matrix(self, filepath: str):
        """Load AU-EMO matrix"""
        self.au_emo_matrix.load(filepath)

    def regularize_au_emo_matrix(self, strength: float = 0.01):
        """Regularize AU-EMO matrix towards prior"""
        return self.au_emo_matrix.regularize_to_prior(strength)


class SingleModalityAUEmotionNetwork(nn.Module):
    """
    Single-modality version of AU-Emotion Network

    Used for multimodal consistency checking.
    Only processes one modality at a time.

    Parameters:
    -----------
    modality : str
        Which modality to use ('text', 'audio', or 'video')
    base_network : AUEmotionNetwork
        Base network to extract parameters from
    """

    def __init__(self, modality: str, base_network: AUEmotionNetwork):
        super().__init__()

        assert modality in ['text', 'audio', 'video'], \
            f"Invalid modality: {modality}"

        self.modality = modality
        self.num_aus = base_network.num_aus
        self.num_emotions = base_network.num_emotions

        # Share encoder and hypergraph from base network
        if modality == 'text':
            self.encoder = base_network.text_encoder
        elif modality == 'audio':
            self.encoder = base_network.audio_encoder
        else:  # video
            self.encoder = base_network.video_encoder

        # Share AU predictor and AU-EMO matrix
        self.au_predictor = base_network.au_predictor
        self.au_emo_matrix = base_network.au_emo_matrix

        # Single-modality classifier
        hidden_dim = base_network.hypergraph.hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_emotions)
        )

    def forward(
        self,
        features: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with single modality

        Args:
            features: [batch_size, T, feature_dim]
            masks: [batch_size, T]

        Returns:
            dict with emotion predictions
        """
        # Encode
        encoded = self.encoder(features, mask=masks)

        # Pool
        if masks is not None:
            mask_expanded = masks.unsqueeze(-1).float()
            valid_counts = masks.sum(dim=1, keepdim=True).float()
            pooled = (encoded * mask_expanded).sum(dim=1) / valid_counts
        else:
            pooled = encoded.mean(dim=1)

        # Predict
        emo_logits = self.classifier(pooled)

        return {
            'emo_logits': emo_logits,
            'pooled_features': pooled
        }

    def predict(
        self,
        features: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict emotion labels

        Returns:
            predictions: [batch_size]
            confidence: [batch_size]
        """
        with torch.no_grad():
            output = self.forward(features, masks)
            probs = F.softmax(output['emo_logits'], dim=1)
            predictions = torch.argmax(probs, dim=1)
            confidence = torch.max(probs, dim=1)[0]

        return predictions, confidence


if __name__ == "__main__":
    # Test the network
    print("Testing AU-Emotion Network...")

    batch_size = 4
    T = 50
    text_dim, audio_dim, video_dim = 768, 768, 768
    num_aus = 23
    num_emotions = 6

    # Create test batch
    text_features = torch.randn(batch_size, T, text_dim)
    audio_features = torch.randn(batch_size, T, audio_dim)
    video_features = torch.randn(batch_size, T, video_dim)
    masks = torch.ones(batch_size, T, dtype=torch.bool)
    labels = torch.randint(0, num_emotions, (batch_size,))

    # Simulate variable-length sequences
    masks[0, 30:] = False
    masks[1, 40:] = False
    masks[3, 25:] = False

    # Create network
    network = AUEmotionNetwork(
        text_input_dim=text_dim,
        audio_input_dim=audio_dim,
        video_input_dim=video_dim,
        num_aus=num_aus,
        num_emotions=num_emotions,
        device='cpu'
    )

    # Forward pass
    output = network(text_features, audio_features, video_features, masks, labels)

    print("\nOutput shapes:")
    print(f"  au_probs: {output['au_probs'].shape}")
    print(f"  emo_from_au: {output['emo_from_au'].shape}")
    print(f"  emo_direct: {output['emo_direct'].shape}")
    print(f"  au_emo_matrix: {output['au_emo_matrix'].shape}")

    print("\nLosses:")
    print(f"  total_loss: {output['loss'].item():.4f}")
    print(f"  cls_loss: {output['cls_loss'].item():.4f}")
    print(f"  cls_loss_au: {output['cls_loss_au'].item():.4f}")
    print(f"  cls_loss_direct: {output['cls_loss_direct'].item():.4f}")

    # Test predictions
    pred_au, conf_au = network.predict_from_au(text_features, audio_features, video_features, masks)
    pred_direct, conf_direct = network.predict_direct(text_features, audio_features, video_features, masks)

    print("\nPredictions:")
    print(f"  From AU: {pred_au}, confidence: {conf_au}")
    print(f"  Direct: {pred_direct}, confidence: {conf_direct}")
    print(f"  True labels: {labels}")

    # Test AU-EMO matrix update
    stats = network.update_au_emo_matrix(
        output['au_probs'],
        labels,
        is_seen=True
    )
    print(f"\nAU-EMO update stats: {stats}")

    print("\n✓ All tests passed!")
