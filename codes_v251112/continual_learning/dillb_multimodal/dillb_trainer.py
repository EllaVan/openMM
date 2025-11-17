"""
DILLB-Style Domain Incremental Learning Trainer

Implements complete training pipeline inspired by DILLB for multimodal
emotion recognition with domain incremental learning.

Key Features:
1. Multi-head architecture (shared backbone + domain-specific heads)
2. Knowledge distillation from previous tasks
3. Optional backbone freezing after Task 0
4. Combined with EWC for stronger anti-forgetting
5. Multimodal consistency checking for unseen classes

Training Strategy:
------------------
Task 0 (Source domain):
  1. Train entire network from scratch
  2. No distillation (no previous tasks)
  3. Save as first teacher

Task 1+ (Target domains):
  1. Add new domain heads
  2. Optionally freeze backbone
  3. Train with:
     - Task loss (current domain labels)
     - Distillation loss (from all previous teachers)
     - EWC loss (optional)
  4. Save as teacher for future tasks

References:
- DILLB: https://github.com/Disguiser15/DILLB
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import json
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from multi_head_network import MultiHeadMultimodalNetwork
from knowledge_distillation import MultiTaskDistillationManager
from continual_learning.ewc import OnlineEWC
from continual_learning.consistency_checker import MultimodalConsistencyChecker, ConsistencyStrategy


class DILLBTrainer:
    """
    DILLB-Style Trainer for Domain Incremental Learning

    Orchestrates multi-head training with knowledge distillation
    """

    def __init__(
        self,
        model: MultiHeadMultimodalNetwork,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        # Knowledge distillation settings
        use_distillation: bool = True,
        kd_temperature: float = 2.0,
        alpha_kd: float = 0.3,
        alpha_feature: float = 0.2,
        alpha_au: float = 0.1,
        # EWC settings
        use_ewc: bool = True,
        ewc_lambda: float = 1000.0,
        # Consistency checking
        consistency_strategy: ConsistencyStrategy = ConsistencyStrategy.MAJORITY,
        min_confidence: float = 0.8,
        # Loss weights
        task_loss_weight: float = 1.0,
        aux_loss_weight: float = 0.1,  # Direct classifier auxiliary loss
        # Saving
        save_dir: str = './checkpoints/dillb'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

        # Knowledge distillation
        self.use_distillation = use_distillation
        if use_distillation:
            self.distillation_manager = MultiTaskDistillationManager(
                temperature=kd_temperature,
                alpha_kd=alpha_kd,
                alpha_feature=alpha_feature,
                alpha_au=alpha_au,
                device=device
            )
        else:
            self.distillation_manager = None

        # EWC
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda
        if use_ewc:
            self.ewc = OnlineEWC(model, device=device)
        else:
            self.ewc = None

        # Consistency checking
        self.consistency_checker = MultimodalConsistencyChecker(
            model=model,
            strategy=consistency_strategy,
            device=device
        )
        self.min_confidence = min_confidence

        # Loss weights
        self.task_loss_weight = task_loss_weight
        self.aux_loss_weight = aux_loss_weight

        # Directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.training_stats = {
            'tasks': [],
            'distillation_history': []
        }

    def train_task(
        self,
        task_id: int,
        task_name: str,
        domain_id: str,
        seen_loader: torch.utils.data.DataLoader,
        unseen_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 10,
        freeze_backbone: bool = False,
        log_interval: int = 10
    ) -> Dict:
        """
        Train on a single task/domain

        Args:
            task_id: Task identifier
            task_name: Task name
            domain_id: Domain identifier (for multi-head)
            seen_loader: Seen class samples with labels
            unseen_loader: Unseen class samples (optional)
            num_epochs: Training epochs
            freeze_backbone: Whether to freeze backbone (after Task 0)
            log_interval: Logging interval

        Returns:
            task_stats: Training statistics
        """
        print(f"\n{'='*80}")
        print(f"Training Task {task_id}: {task_name} (Domain: {domain_id})")
        print(f"{'='*80}")

        # Register new domain
        self.model.add_domain(domain_id)

        # Optionally freeze backbone
        if freeze_backbone and task_id > 0:
            self.model._freeze_backbone()

        task_stats = {
            'task_id': task_id,
            'task_name': task_name,
            'domain_id': domain_id,
            'epochs': [],
            'seen_metrics': []
        }

        # Train on seen classes
        print(f"\n[Task {task_id}] Training on Seen Classes")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            epoch_stats = self._train_seen_epoch(
                seen_loader=seen_loader,
                domain_id=domain_id,
                current_task_id=task_id,
                log_interval=log_interval
            )

            print(f"  Task loss: {epoch_stats['task_loss']:.4f}")
            if self.use_distillation and task_id > 0:
                print(f"  Distill loss: {epoch_stats['distill_loss']:.4f}")
            if self.use_ewc and task_id > 0:
                print(f"  EWC loss: {epoch_stats['ewc_loss']:.4f}")

            # Evaluate
            eval_stats = self._evaluate(seen_loader, domain_id)
            task_stats['epochs'].append(epoch)
            task_stats['seen_metrics'].append(eval_stats)

            print(f"  Accuracy: {eval_stats['accuracy']:.4f}")

        # Train on unseen classes (if provided)
        if unseen_loader is not None:
            print(f"\n[Task {task_id}] Training on Unseen Classes")
            unseen_stats = self._train_unseen(
                unseen_loader=unseen_loader,
                domain_id=domain_id,
                num_epochs=num_epochs // 2
            )
            task_stats['unseen_metrics'] = unseen_stats

        # Save task as teacher for future distillation
        if self.use_distillation:
            self.distillation_manager.add_teacher(domain_id, self.model)

        # EWC consolidation
        if self.use_ewc and task_id >= 0:
            print(f"\n[Task {task_id}] EWC Consolidation")
            self.ewc.consolidate(seen_loader)

        # Save checkpoint
        self._save_checkpoint(task_id, task_name, domain_id)

        self.training_stats['tasks'].append(task_stats)

        return task_stats

    def _train_seen_epoch(
        self,
        seen_loader: torch.utils.data.DataLoader,
        domain_id: str,
        current_task_id: int,
        log_interval: int = 10
    ) -> Dict:
        """Train one epoch on seen classes"""
        self.model.train()

        total_task_loss = 0
        total_distill_loss = 0
        total_ewc_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(seen_loader):
            # Move to device
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)
            masks = batch.get('masks')

            # Forward pass
            outputs = self.model(
                text, audio, video, masks,
                domain_id=domain_id,
                return_features=True
            )

            # 1. Task loss (current domain)
            loss_emo_au = F.cross_entropy(outputs['emo_from_au'], labels)
            loss_emo_direct = F.cross_entropy(outputs['emo_direct'], labels)
            task_loss = loss_emo_au + self.aux_loss_weight * loss_emo_direct

            # Total loss
            loss = self.task_loss_weight * task_loss

            # 2. Knowledge distillation loss (from previous tasks)
            distill_loss = 0.0
            if self.use_distillation and current_task_id > 0:
                distill_outputs = self.distillation_manager.compute_distillation_loss(
                    student_model=self.model,
                    text=text,
                    audio=audio,
                    video=video,
                    masks=masks,
                    current_task_id=domain_id
                )
                distill_loss = distill_outputs['total_distill_loss']
                loss += distill_loss

            # 3. EWC loss (anti-forgetting)
            ewc_loss = 0.0
            if self.use_ewc and current_task_id > 0 and self.ewc.is_consolidated:
                ewc_loss = self.ewc.penalty()
                loss += self.ewc_lambda * ewc_loss

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_task_loss += task_loss.item()
            total_distill_loss += distill_loss.item() if isinstance(distill_loss, torch.Tensor) else distill_loss
            total_ewc_loss += ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else ewc_loss
            num_batches += 1

            if (batch_idx + 1) % log_interval == 0:
                print(f"    Batch {batch_idx+1}/{len(seen_loader)}, "
                      f"Loss: {loss.item():.4f}")

        return {
            'task_loss': total_task_loss / num_batches,
            'distill_loss': total_distill_loss / num_batches,
            'ewc_loss': total_ewc_loss / num_batches
        }

    def _train_unseen(
        self,
        unseen_loader: torch.utils.data.DataLoader,
        domain_id: str,
        num_epochs: int = 5
    ) -> Dict:
        """Train on unseen classes with consistency checking"""
        print("  Training on unseen classes (consistency checking)...")

        stats_per_epoch = []

        for epoch in range(num_epochs):
            # Collect pseudo-labels
            pseudo_data = self._collect_pseudo_labels(unseen_loader, domain_id)

            if pseudo_data['num_consistent'] == 0:
                print(f"    Epoch {epoch+1}: No consistent samples, skipping")
                continue

            print(f"    Epoch {epoch+1}: {pseudo_data['num_consistent']} consistent samples")

            # Train with pseudo-labels (low weight)
            self.model.train()

            text = pseudo_data['text']
            audio = pseudo_data['audio']
            video = pseudo_data['video']
            labels = pseudo_data['pseudo_labels']
            confidences = pseudo_data['confidences']

            # Filter by confidence
            high_conf_mask = confidences >= self.min_confidence
            if not high_conf_mask.any():
                continue

            text = text[high_conf_mask]
            audio = audio[high_conf_mask]
            video = video[high_conf_mask]
            labels = labels[high_conf_mask]
            confidences = confidences[high_conf_mask]

            # Forward pass
            outputs = self.model(text, audio, video, None, domain_id=domain_id)

            # Confidence-weighted loss
            loss = F.cross_entropy(outputs['emo_from_au'], labels, reduction='none')
            loss = (loss * confidences).mean() * 0.3  # Low weight for unseen

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            stats_per_epoch.append({
                'epoch': epoch,
                'consistent_samples': pseudo_data['num_consistent'],
                'loss': loss.item()
            })

        return {'epochs': stats_per_epoch}

    def _collect_pseudo_labels(
        self,
        dataloader: torch.utils.data.DataLoader,
        domain_id: str
    ) -> Dict:
        """Collect pseudo-labels via consistency checking"""
        self.model.eval()

        all_text, all_audio, all_video = [], [], []
        all_pseudo_labels, all_confidences, all_is_consistent = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)

                # Temporarily set domain for consistency checking
                results = self.consistency_checker.check_consistency(
                    text, audio, video, batch.get('masks')
                )

                all_text.append(text)
                all_audio.append(audio)
                all_video.append(video)
                all_pseudo_labels.append(results['consensus_label'])
                all_confidences.append(results['confidence'])
                all_is_consistent.append(results['is_consistent'])

        # Concatenate
        all_text = torch.cat(all_text, dim=0)
        all_audio = torch.cat(all_audio, dim=0)
        all_video = torch.cat(all_video, dim=0)
        all_pseudo_labels = torch.cat(all_pseudo_labels, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        all_is_consistent = torch.cat(all_is_consistent, dim=0)

        # Filter consistent
        mask = all_is_consistent

        return {
            'text': all_text[mask],
            'audio': all_audio[mask],
            'video': all_video[mask],
            'pseudo_labels': all_pseudo_labels[mask],
            'confidences': all_confidences[mask],
            'num_consistent': mask.sum().item(),
            'total_samples': len(all_is_consistent)
        }

    def _evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        domain_id: str
    ) -> Dict:
        """Evaluate on a domain"""
        self.model.eval()

        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(text, audio, video, None, domain_id=domain_id)
                preds = outputs['emo_from_au'].argmax(dim=1)

                all_preds.append(preds)
                all_labels.append(labels)

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        accuracy = (all_preds == all_labels).float().mean().item()

        return {
            'accuracy': accuracy,
            'predictions': all_preds.cpu().numpy(),
            'labels': all_labels.cpu().numpy()
        }

    def _save_checkpoint(self, task_id: int, task_name: str, domain_id: str):
        """Save checkpoint"""
        checkpoint_path = self.save_dir / f'task_{task_id}_checkpoint.pt'

        checkpoint = {
            'task_id': task_id,
            'task_name': task_name,
            'domain_id': domain_id,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'registered_domains': list(self.model.registered_domains)
        }

        if self.use_ewc:
            checkpoint['ewc_state'] = {
                'fisher': self.ewc.fisher_dict,
                'optimal_params': self.ewc.optimal_params_dict
            }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def evaluate_all_tasks(
        self,
        task_configs: List,
        data_splitter,
        batch_size: int = 32
    ) -> Dict:
        """Evaluate on all previous tasks"""
        results = {}

        for task_config in task_configs:
            domain_id = f"task_{task_config.task_id}"

            # Get dataloader
            seen_loader, _ = data_splitter.create_task_dataloaders(
                task_config,
                batch_size=batch_size,
                shuffle=False
            )

            # Evaluate
            eval_stats = self._evaluate(seen_loader, domain_id)
            results[domain_id] = eval_stats

            print(f"Task {task_config.task_id} ({domain_id}): "
                  f"Acc = {eval_stats['accuracy']:.4f}")

        return results


def test_dillb_trainer():
    """Test DILLB trainer"""
    print("Testing DILLB Trainer...")

    # Create model
    model = MultiHeadMultimodalNetwork(
        num_aus=23,
        num_emotions=6,
        device='cpu'
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Create trainer
    trainer = DILLBTrainer(
        model=model,
        optimizer=optimizer,
        device='cpu',
        use_distillation=True,
        use_ewc=True
    )

    print("✓ DILLB trainer initialized successfully!")


if __name__ == "__main__":
    test_dillb_trainer()
