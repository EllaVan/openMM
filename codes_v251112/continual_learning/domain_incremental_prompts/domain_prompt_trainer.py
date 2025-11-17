"""
Domain Incremental Learning with Multimodal Prompts

Comprehensive trainer integrating:
1. S-Prompts style prompt learning
2. UDIL adaptive loss weighting
3. DARE contrastive learning
4. AU-based prototype retrieval

Training Strategy:
==================
Domain 0:
  1. Learn prompts for all modalities
  2. Build AU prototypes
  3. Train with task loss only

Domain 1+:
  1. Add new prompts for this domain
  2. Build AU prototypes for this domain
  3. Train with adaptive multi-loss:
     - Task loss (current domain)
     - Distillation loss (previous domains)
     - Contrastive loss (AU representations)
     - Alignment loss (domain adaptation)
  4. Automatically retrieve prompts at test time

Inference:
==========
1. Extract AU features from input
2. Use K-NN to find nearest domain
3. Apply corresponding domain prompts
4. Forward through network
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys
import copy

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from multimodal_prompts import MultimodalDomainPrompts, AUPrototypeBank
from adaptive_loss_weighting import (
    AdaptiveLossWeighting,
    DomainContrastiveLoss,
    FeatureAlignmentLoss
)
from continual_learning.au_emotion_network import AUEmotionNetwork, AUPredictor
from hyper_fusion.hypergraph_model import HypergraphMultimodalFusion


class PromptedMultimodalNetwork(nn.Module):
    """
    Multimodal network with domain-specific prompts

    Architecture:
    1. Domain prompts prepended to inputs
    2. Multimodal encoder (shared across domains)
    3. AU predictor (shared across domains)
    4. Emotion classifier (shared across domains)
    """

    def __init__(
        self,
        text_input_dim: int = 768,
        audio_input_dim: int = 768,
        video_input_dim: int = 768,
        num_aus: int = 23,
        num_emotions: int = 6,
        prompt_length: int = 5,
        encoder_hidden_dim: int = 256,
        num_hyperedges: int = 64,
        device: str = 'cuda'
    ):
        super().__init__()

        self.device_str = device

        # Domain prompts
        self.domain_prompts = MultimodalDomainPrompts(
            text_prompt_length=prompt_length,
            audio_prompt_length=prompt_length,
            video_prompt_length=prompt_length,
            text_dim=text_input_dim,
            audio_dim=audio_input_dim,
            video_dim=video_input_dim,
            device=device
        )

        # Shared multimodal encoder
        self.encoder = HypergraphMultimodalFusion(
            text_input_dim=text_input_dim,
            audio_input_dim=audio_input_dim,
            video_input_dim=video_input_dim,
            encoder_hidden_dim=encoder_hidden_dim,
            num_hyperedges=num_hyperedges,
            use_bottleneck=True,
            device=device
        )

        # Shared AU predictor
        self.au_predictor = AUPredictor(
            input_dim=768,  # After bottleneck
            num_aus=num_aus,
            device=device
        )

        # Shared emotion classifier
        self.emotion_classifier = nn.Sequential(
            nn.Linear(num_aus, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        ).to(device)

    def add_domain(self, domain_id: str):
        """Register a new domain"""
        self.domain_prompts.add_domain(domain_id)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        domain_id: Optional[str] = None,
        masks: Optional[Dict] = None,
        return_au: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            text: [batch, seq_len, text_dim]
            audio: [batch, seq_len, audio_dim]
            video: [batch, seq_len, video_dim]
            domain_id: Which prompts to use (None = no prompts)
            masks: Optional masks
            return_au: Whether to return AU features

        Returns:
            Dict with predictions
        """
        # Apply domain prompts if specified
        if domain_id is not None:
            prompted = self.domain_prompts(text, audio, video, domain_id)
            text = prompted['text']
            audio = prompted['audio']
            video = prompted['video']

        # Multimodal encoding
        multimodal_features = self.encoder(text, audio, video, masks)
        # [batch, 768]

        # AU prediction
        au_probs = self.au_predictor(multimodal_features)
        # [batch, num_aus]

        # Emotion prediction
        emo_logits = self.emotion_classifier(au_probs)
        # [batch, num_emotions]

        outputs = {
            'emo_logits': emo_logits,
            'emo_probs': F.softmax(emo_logits, dim=1)
        }

        if return_au:
            outputs['au_probs'] = au_probs
            outputs['multimodal_features'] = multimodal_features

        return outputs


class DomainPromptTrainer:
    """
    Trainer for Domain Incremental Learning with Prompts
    """

    def __init__(
        self,
        model: PromptedMultimodalNetwork,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        # Adaptive weighting
        use_adaptive_weighting: bool = True,
        # Contrastive learning
        use_contrastive: bool = True,
        contrastive_temperature: float = 0.07,
        # Feature alignment
        use_alignment: bool = True,
        alignment_method: str = 'mmd',
        # AU prototype
        num_prototypes_per_domain: int = 10,
        # Distillation
        distillation_temperature: float = 2.0,
        # Saving
        save_dir: str = './checkpoints/domain_prompts'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

        # AU prototype bank
        self.prototype_bank = AUPrototypeBank(
            num_prototypes_per_domain=num_prototypes_per_domain,
            device=device
        )

        # Adaptive loss weighting
        self.use_adaptive_weighting = use_adaptive_weighting
        if use_adaptive_weighting:
            self.loss_weighting = AdaptiveLossWeighting(
                num_losses=4,
                learnable=True,
                device=device
            )
            # Add to optimizer
            self.optimizer.add_param_group({
                'params': [self.loss_weighting.logits],
                'lr': 0.01
            })
        else:
            self.loss_weighting = None

        # Contrastive loss
        self.use_contrastive = use_contrastive
        if use_contrastive:
            self.contrastive_loss = DomainContrastiveLoss(
                temperature=contrastive_temperature
            )

        # Feature alignment
        self.use_alignment = use_alignment
        if use_alignment:
            self.alignment_loss = FeatureAlignmentLoss(
                method=alignment_method
            )

        # Distillation
        self.distillation_temperature = distillation_temperature
        self.teachers = {}  # domain_id -> teacher model

        # Directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.training_stats = []

    def train_domain(
        self,
        domain_id: str,
        domain_name: str,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 10,
        is_first_domain: bool = False
    ) -> Dict:
        """
        Train on a single domain

        Args:
            domain_id: Domain identifier
            domain_name: Domain name
            train_loader: Training data
            val_loader: Validation data
            num_epochs: Number of epochs
            is_first_domain: Whether this is the first domain

        Returns:
            Training statistics
        """
        print(f"\n{'='*80}")
        print(f"Training Domain: {domain_name} (ID: {domain_id})")
        print(f"First domain: {is_first_domain}")
        print(f"{'='*80}")

        # Register domain
        self.model.add_domain(domain_id)

        domain_stats = {
            'domain_id': domain_id,
            'domain_name': domain_name,
            'epochs': []
        }

        # Train epochs
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            epoch_stats = self._train_epoch(
                train_loader=train_loader,
                domain_id=domain_id,
                is_first_domain=is_first_domain
            )

            # Evaluate
            if val_loader is not None:
                val_stats = self._evaluate(val_loader, domain_id)
                epoch_stats['val_accuracy'] = val_stats['accuracy']
                print(f"  Validation accuracy: {val_stats['accuracy']:.4f}")

            domain_stats['epochs'].append(epoch_stats)

        # Build AU prototypes for this domain
        print(f"\nBuilding AU prototypes for domain: {domain_id}")
        self._build_prototypes(train_loader, domain_id)

        # Save as teacher for future distillation
        self.teachers[domain_id] = copy.deepcopy(self.model)
        self.teachers[domain_id].eval()
        for param in self.teachers[domain_id].parameters():
            param.requires_grad = False

        print(f"Saved teacher model for domain: {domain_id}")

        # Save checkpoint
        self._save_checkpoint(domain_id)

        self.training_stats.append(domain_stats)

        return domain_stats

    def _train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        domain_id: str,
        is_first_domain: bool
    ) -> Dict:
        """Train one epoch"""
        self.model.train()

        total_task_loss = 0
        total_distill_loss = 0
        total_contrastive_loss = 0
        total_alignment_loss = 0
        correct = 0
        total = 0

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            outputs = self.model(
                text, audio, video,
                domain_id=domain_id,
                masks=batch.get('masks'),
                return_au=True
            )

            # 1. Task loss
            task_loss = F.cross_entropy(outputs['emo_logits'], labels)

            losses = {'task': task_loss}

            # 2. Distillation loss (if not first domain)
            if not is_first_domain:
                distill_loss = self._compute_distillation_loss(
                    text, audio, video, batch.get('masks'),
                    outputs, domain_id
                )
                losses['distillation'] = distill_loss
                total_distill_loss += distill_loss.item()

            # 3. Contrastive loss
            if self.use_contrastive and len(self.teachers) > 0:
                # Create domain IDs tensor
                domain_ids = torch.full_like(labels, int(domain_id.split('_')[-1]))

                contrastive_loss = self.contrastive_loss(
                    outputs['au_probs'], labels, domain_ids
                )
                losses['contrastive'] = contrastive_loss
                total_contrastive_loss += contrastive_loss.item()

            # 4. Alignment loss
            if self.use_alignment and not is_first_domain:
                alignment_loss = self._compute_alignment_loss(outputs['au_probs'])
                losses['alignment'] = alignment_loss
                total_alignment_loss += alignment_loss.item()

            # Combine losses
            if self.use_adaptive_weighting and not is_first_domain:
                total_loss = self.loss_weighting(losses)
            else:
                # Simple weighted sum for first domain
                total_loss = losses['task']
                if 'distillation' in losses:
                    total_loss += 0.5 * losses['distillation']
                if 'contrastive' in losses:
                    total_loss += 0.3 * losses['contrastive']
                if 'alignment' in losses:
                    total_loss += 0.2 * losses['alignment']

            # Backward
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Statistics
            total_task_loss += task_loss.item()
            preds = outputs['emo_logits'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Epoch statistics
        stats = {
            'task_loss': total_task_loss / len(train_loader),
            'accuracy': correct / total
        }

        if not is_first_domain:
            stats['distill_loss'] = total_distill_loss / len(train_loader)
            stats['contrastive_loss'] = total_contrastive_loss / len(train_loader)
            stats['alignment_loss'] = total_alignment_loss / len(train_loader)

            # Log adaptive weights
            if self.use_adaptive_weighting:
                weights = self.loss_weighting.get_weights()
                stats['adaptive_weights'] = weights.detach().cpu().numpy()

        print(f"  Task loss: {stats['task_loss']:.4f}, Accuracy: {stats['accuracy']:.4f}")

        return stats

    def _compute_distillation_loss(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        masks: Optional[Dict],
        student_outputs: Dict,
        current_domain_id: str
    ) -> torch.Tensor:
        """Compute distillation loss from all previous teachers"""
        if len(self.teachers) == 0:
            return torch.tensor(0.0, device=self.device)

        T = self.distillation_temperature
        distill_loss = 0.0

        for teacher_domain_id, teacher in self.teachers.items():
            if teacher_domain_id == current_domain_id:
                continue

            with torch.no_grad():
                teacher_outputs = teacher(
                    text, audio, video,
                    domain_id=teacher_domain_id,
                    masks=masks
                )

            # KL divergence
            student_soft = F.log_softmax(student_outputs['emo_logits'] / T, dim=1)
            teacher_soft = F.softmax(teacher_outputs['emo_logits'] / T, dim=1)

            kl = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
            distill_loss += kl

        # Average over teachers
        distill_loss = distill_loss / max(len(self.teachers), 1)

        return distill_loss

    def _compute_alignment_loss(self, au_features: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss with previous domain AU features"""
        # For simplicity, use within-batch alignment
        # In practice, could use memory buffer
        batch_size = au_features.shape[0]
        half = batch_size // 2

        if half < 2:
            return torch.tensor(0.0, device=self.device)

        source = au_features[:half]
        target = au_features[half:]

        return self.alignment_loss(source, target)

    def _build_prototypes(
        self,
        dataloader: torch.utils.data.DataLoader,
        domain_id: str
    ):
        """Build AU prototypes for a domain"""
        self.model.eval()

        all_au_features = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    text, audio, video,
                    domain_id=domain_id,
                    masks=batch.get('masks'),
                    return_au=True
                )

                all_au_features.append(outputs['au_probs'])
                all_labels.append(labels)

        all_au_features = torch.cat(all_au_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Build prototypes
        self.prototype_bank.build_prototypes(domain_id, all_au_features, all_labels)

    def infer_with_auto_prompt(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        masks: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, str]:
        """
        Inference with automatic prompt retrieval

        Args:
            text, audio, video: Input features
            masks: Optional masks

        Returns:
            predictions: Predicted emotion logits
            retrieved_domain: Retrieved domain ID
        """
        self.model.eval()

        with torch.no_grad():
            # First, get AU features without prompts
            outputs_no_prompt = self.model(
                text, audio, video,
                domain_id=None,
                masks=masks,
                return_au=True
            )

            # Retrieve domain based on AU features
            retrieved_domain = self.prototype_bank.retrieve_domain(
                outputs_no_prompt['au_probs']
            )

            # Forward with retrieved prompts
            final_outputs = self.model(
                text, audio, video,
                domain_id=retrieved_domain,
                masks=masks
            )

        return final_outputs['emo_logits'], retrieved_domain

    def _evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        domain_id: str
    ) -> Dict:
        """Evaluate on a domain"""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(
                    text, audio, video,
                    domain_id=domain_id,
                    masks=batch.get('masks')
                )

                preds = outputs['emo_logits'].argmax(dim=1)
                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        accuracy = (all_preds == all_labels).float().mean().item()

        return {'accuracy': accuracy}

    def _save_checkpoint(self, domain_id: str):
        """Save checkpoint"""
        checkpoint_path = self.save_dir / f'domain_{domain_id}_checkpoint.pt'

        checkpoint = {
            'domain_id': domain_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }

        if self.use_adaptive_weighting:
            checkpoint['loss_weights'] = self.loss_weighting.get_weights()

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    print("Domain Prompt Trainer module")
    print("Use domain_prompt_main.py for full training")
