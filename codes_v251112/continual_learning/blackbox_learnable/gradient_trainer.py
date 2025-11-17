"""
Gradient Descent Trainer for Blackbox Learnable AU-EMO Matrix

Implements end-to-end gradient descent training where the AU-EMO matrix
is optimized jointly with the neural network via backpropagation.

Simpler than EM approach - single optimization pass updates everything.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
import json
import sys

# Add parent directory for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from learnable_matrix import LearnableAUEMOMatrix
from continual_learning.au_emotion_network import AUEmotionNetwork
from continual_learning.consistency_checker import MultimodalConsistencyChecker, ConsistencyStrategy
from continual_learning.ewc import OnlineEWC


class GradientTrainerBlackbox:
    """
    Gradient Descent Trainer for Blackbox Continual Learning

    Training Flow:
    --------------
    Task 0:
        1. Warmup: Train entire network including matrix (few epochs)
        2. Seen: Continue training with regularization
        3. Unseen: Consistency checking + low-weight updates

    Task 1 to T:
        1. Seen: Train with EWC penalty + matrix regularization
        2. Unseen: Consistency checking + low-weight updates
        3. Consolidate EWC

    Key Differences from Whitebox EM:
    ----------------------------------
    - Single optimization pass (no E-step/M-step)
    - Matrix updated via gradient descent (no Bayesian update)
    - Matrix regularization via KL divergence to prior
    - Simpler and faster training
    - Less interpretable

    Parameters:
    -----------
    model : AUEmotionNetwork
        Multimodal network with AU predictor
    au_emo_matrix : LearnableAUEMOMatrix
        Learnable AU-EMO matrix
    optimizer : torch.optim.Optimizer
        Optimizer for all parameters (network + matrix)
    device : str
        Device for training
    use_ewc : bool
        Whether to use EWC for anti-forgetting
    ewc_lambda : float
        EWC regularization strength
    matrix_reg_lambda : float
        Matrix regularization strength (KL to prior)
    consistency_strategy : ConsistencyStrategy
        Strategy for multimodal consistency checking
    min_confidence : float
        Minimum confidence for unseen updates
    seen_loss_weight : float
        Weight for seen class loss
    unseen_loss_weight : float
        Weight for unseen class loss (lower for pseudo-labels)
    save_dir : str
        Directory to save checkpoints
    """

    def __init__(
        self,
        model: AUEmotionNetwork,
        au_emo_matrix: LearnableAUEMOMatrix,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        use_ewc: bool = True,
        ewc_lambda: float = 1000.0,
        matrix_reg_lambda: float = 0.1,
        consistency_strategy: ConsistencyStrategy = ConsistencyStrategy.MAJORITY,
        min_confidence: float = 0.8,
        seen_loss_weight: float = 1.0,
        unseen_loss_weight: float = 0.3,
        save_dir: str = './checkpoints/blackbox'
    ):
        self.model = model.to(device)
        self.au_emo_matrix = au_emo_matrix
        self.optimizer = optimizer
        self.device = device

        # Continual learning settings
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda
        self.ewc = OnlineEWC(model, device=device) if use_ewc else None

        # Matrix regularization
        self.matrix_reg_lambda = matrix_reg_lambda

        # Consistency checking
        self.consistency_checker = MultimodalConsistencyChecker(
            model=model,
            strategy=consistency_strategy,
            device=device
        )
        self.min_confidence = min_confidence

        # Loss weights
        self.seen_loss_weight = seen_loss_weight
        self.unseen_loss_weight = unseen_loss_weight

        # Directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.training_stats = {
            'tasks': [],
            'matrix_evolution': []
        }

    def train_task(
        self,
        task_id: int,
        task_name: str,
        seen_loader: torch.utils.data.DataLoader,
        unseen_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 10,
        log_interval: int = 10
    ) -> Dict:
        """
        Train on a single task using gradient descent

        Parameters:
        -----------
        task_id : int
            Task identifier
        task_name : str
            Task name
        seen_loader : DataLoader
            Seen class samples with labels
        unseen_loader : DataLoader, optional
            Unseen class samples without labels
        num_epochs : int
            Total epochs for this task
        log_interval : int
            Logging interval (batches)

        Returns:
        --------
        task_stats : dict
            Training statistics for this task
        """
        print(f"\n{'='*80}")
        print(f"Training Task {task_id}: {task_name}")
        print(f"{'='*80}")

        task_stats = {
            'task_id': task_id,
            'task_name': task_name,
            'epochs': [],
            'seen_metrics': [],
            'unseen_metrics': []
        }

        # Phase 1: Warmup (Task 0 only)
        if task_id == 0:
            print(f"\n[Task {task_id}] Phase 1: Warmup")
            self._warmup_phase(seen_loader, num_epochs=3, log_interval=log_interval)

        # Phase 2: Seen Training
        print(f"\n[Task {task_id}] Phase 2: Seen Training")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Train on seen classes
            seen_stats = self._train_seen_epoch(seen_loader, log_interval=log_interval)

            print(f"  Seen training: loss={seen_stats['avg_loss']:.4f}, "
                  f"acc={seen_stats['accuracy']:.4f}")

            # Evaluate
            eval_stats = self._evaluate_seen(seen_loader)
            task_stats['epochs'].append(epoch)
            task_stats['seen_metrics'].append(eval_stats)

            print(f"  Seen evaluation: acc={eval_stats['accuracy']:.4f}")

            # Track matrix evolution
            matrix_stats = self.au_emo_matrix.get_statistics()
            self.training_stats['matrix_evolution'].append({
                'task_id': task_id,
                'epoch': epoch,
                'kl_from_prior': matrix_stats['kl_from_prior']
            })

        # Phase 3: Unseen Training (if available)
        if unseen_loader is not None:
            print(f"\n[Task {task_id}] Phase 3: Unseen Training")
            unseen_stats = self._train_unseen(
                unseen_loader,
                num_epochs=num_epochs // 2,
                log_interval=log_interval
            )
            task_stats['unseen_metrics'] = unseen_stats

        # Phase 4: EWC Consolidation
        if self.use_ewc and task_id >= 0:
            print(f"\n[Task {task_id}] Phase 4: EWC Consolidation")
            self.ewc.consolidate(seen_loader)
            print(f"  Fisher information consolidated")

        # Save task checkpoint
        self._save_task_checkpoint(task_id, task_name)

        # Update training statistics
        self.training_stats['tasks'].append(task_stats)

        return task_stats

    def _warmup_phase(
        self,
        seen_loader: torch.utils.data.DataLoader,
        num_epochs: int = 3,
        log_interval: int = 10
    ):
        """
        Warmup phase: Train entire network including matrix

        Goal: Initialize all parameters to reasonable values
        """
        print("Warmup: Training entire network...")

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0

            for batch_idx, batch in enumerate(seen_loader):
                # Move to device
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(text, audio, video, batch.get('masks'))

                # Loss: emotion classification via AU path + direct path
                loss_au_path = F.cross_entropy(outputs['emo_from_au'], labels)
                loss_direct = F.cross_entropy(outputs['emo_direct'], labels)

                loss = loss_au_path + 0.1 * loss_direct

                # Light matrix regularization
                loss += 0.01 * self.au_emo_matrix.compute_regularization_loss()

                # Backward and optimize
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % log_interval == 0:
                    print(f"  Epoch {epoch+1}/{num_epochs}, "
                          f"Batch {batch_idx+1}/{len(seen_loader)}, "
                          f"Loss: {loss.item():.4f}")

            avg_loss = total_loss / num_batches
            print(f"Warmup Epoch {epoch+1}/{num_epochs}: avg_loss={avg_loss:.4f}")

    def _train_seen_epoch(
        self,
        dataloader: torch.utils.data.DataLoader,
        log_interval: int = 10
    ) -> Dict:
        """
        Train one epoch on seen classes

        Returns:
        --------
        stats : dict
            Training statistics
        """
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            outputs = self.model(text, audio, video, batch.get('masks'))

            # Primary loss: emotion classification via AU-EMO matrix
            loss_emo = F.cross_entropy(outputs['emo_from_au'], labels)

            # Auxiliary loss: direct path
            loss_direct = F.cross_entropy(outputs['emo_direct'], labels)

            # Matrix regularization: KL divergence to prior
            loss_matrix_reg = self.au_emo_matrix.compute_regularization_loss()

            # Combined loss
            loss = (
                self.seen_loss_weight * loss_emo +
                0.1 * loss_direct +
                self.matrix_reg_lambda * loss_matrix_reg
            )

            # Add EWC penalty if applicable
            if self.use_ewc and self.ewc.is_consolidated:
                loss += self.ewc_lambda * self.ewc.penalty()

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            num_batches += 1

            preds = outputs['emo_from_au'].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return {
            'avg_loss': total_loss / num_batches,
            'accuracy': correct / total,
            'num_batches': num_batches
        }

    def _train_unseen(
        self,
        unseen_loader: torch.utils.data.DataLoader,
        num_epochs: int = 5,
        log_interval: int = 10
    ) -> Dict:
        """
        Train on unseen classes with consistency checking

        Strategy:
        1. Use consistency checker to get pseudo-labels
        2. Train with low weight on high-confidence predictions
        3. Maintain matrix regularization

        Returns:
        --------
        stats : dict
            Unseen training statistics
        """
        print("Unseen training with consistency checking...")

        stats_per_epoch = []

        for epoch in range(num_epochs):
            print(f"\n  Unseen Epoch {epoch+1}/{num_epochs}")

            # Collect pseudo-labels
            pseudo_labels_data = self._collect_pseudo_labels(unseen_loader)

            if pseudo_labels_data['num_consistent'] == 0:
                print(f"    No consistent samples found, skipping epoch")
                continue

            print(f"    Consistent samples: {pseudo_labels_data['num_consistent']}/{pseudo_labels_data['total_samples']} "
                  f"({100*pseudo_labels_data['consistency_rate']:.1f}%)")

            # Train with pseudo-labels
            train_stats = self._train_with_pseudo_labels(
                pseudo_labels_data,
                log_interval=log_interval
            )

            print(f"    Training loss: {train_stats['avg_loss']:.4f}")

            stats_per_epoch.append({
                'epoch': epoch,
                'consistency_rate': pseudo_labels_data['consistency_rate'],
                'avg_loss': train_stats['avg_loss']
            })

        return {
            'epochs': stats_per_epoch,
            'final_consistency_rate': stats_per_epoch[-1]['consistency_rate'] if stats_per_epoch else 0.0
        }

    def _collect_pseudo_labels(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict:
        """
        Collect pseudo-labels using multimodal consistency checking

        Returns:
        --------
        pseudo_labels_data : dict
            Contains batch data, pseudo-labels, and confidences
        """
        self.model.eval()

        all_text = []
        all_audio = []
        all_video = []
        all_masks = []
        all_pseudo_labels = []
        all_confidences = []
        all_is_consistent = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                masks = batch.get('masks')

                # Consistency checking
                results = self.consistency_checker.check_consistency(
                    text, audio, video, masks
                )

                all_text.append(text)
                all_audio.append(audio)
                all_video.append(video)
                all_masks.append(masks)
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

        # Filter to consistent samples only
        consistent_mask = all_is_consistent

        return {
            'text': all_text[consistent_mask],
            'audio': all_audio[consistent_mask],
            'video': all_video[consistent_mask],
            'masks': all_masks,  # Handle separately
            'pseudo_labels': all_pseudo_labels[consistent_mask],
            'confidences': all_confidences[consistent_mask],
            'num_consistent': consistent_mask.sum().item(),
            'total_samples': len(all_is_consistent),
            'consistency_rate': consistent_mask.float().mean().item()
        }

    def _train_with_pseudo_labels(
        self,
        pseudo_labels_data: Dict,
        log_interval: int = 10
    ) -> Dict:
        """
        Train with collected pseudo-labels

        Uses lower loss weight to account for label uncertainty
        """
        if pseudo_labels_data['num_consistent'] == 0:
            return {'avg_loss': 0.0}

        self.model.train()

        # Create mini-batches from pseudo-labeled data
        text = pseudo_labels_data['text']
        audio = pseudo_labels_data['audio']
        video = pseudo_labels_data['video']
        labels = pseudo_labels_data['pseudo_labels']
        confidences = pseudo_labels_data['confidences']

        # Filter by confidence threshold
        high_conf_mask = confidences >= self.min_confidence

        if not high_conf_mask.any():
            return {'avg_loss': 0.0}

        text = text[high_conf_mask]
        audio = audio[high_conf_mask]
        video = video[high_conf_mask]
        labels = labels[high_conf_mask]
        confidences = confidences[high_conf_mask]

        # Forward pass
        outputs = self.model(text, audio, video, None)

        # Loss with confidence weighting
        loss_emo = F.cross_entropy(outputs['emo_from_au'], labels, reduction='none')
        loss_emo = (loss_emo * confidences).mean()  # Weight by confidence

        # Matrix regularization (important for unseen)
        loss_matrix_reg = self.au_emo_matrix.compute_regularization_loss()

        # Combined loss with low weight for unseen
        loss = (
            self.unseen_loss_weight * loss_emo +
            self.matrix_reg_lambda * loss_matrix_reg
        )

        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            'avg_loss': loss.item(),
            'num_samples': high_conf_mask.sum().item()
        }

    def _evaluate_seen(self, dataloader: torch.utils.data.DataLoader) -> Dict:
        """Evaluate on seen classes"""
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(text, audio, video, batch.get('masks'))
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

    def _save_task_checkpoint(self, task_id: int, task_name: str):
        """Save checkpoint after task completion"""
        checkpoint_path = self.save_dir / f'task_{task_id}_checkpoint.pt'

        checkpoint = {
            'task_id': task_id,
            'task_name': task_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'matrix_statistics': self.au_emo_matrix.get_statistics()
        }

        if self.use_ewc:
            checkpoint['ewc_state'] = {
                'fisher': self.ewc.fisher_dict,
                'optimal_params': self.ewc.optimal_params_dict
            }

        torch.save(checkpoint, checkpoint_path)

        # Save matrix separately
        matrix_path = self.save_dir / f'task_{task_id}_matrix.npz'
        self.au_emo_matrix.save(str(matrix_path))

        print(f"Checkpoint saved to {checkpoint_path}")

    def save_final_model(self, filename: str = 'final_model.pt'):
        """Save final model and matrix"""
        final_path = self.save_dir / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'matrix_statistics': self.au_emo_matrix.get_statistics()
        }

        torch.save(checkpoint, final_path)

        # Save final matrix
        matrix_path = self.save_dir / 'final_matrix.npz'
        self.au_emo_matrix.save(str(matrix_path))

        print(f"Final model saved to {final_path}")

    def get_training_summary(self) -> str:
        """Get training summary"""
        summary = []
        summary.append("\n" + "="*80)
        summary.append("BLACKBOX TRAINING SUMMARY")
        summary.append("="*80)

        summary.append(f"\nTotal tasks trained: {len(self.training_stats['tasks'])}")

        summary.append("\nAU-EMO Matrix Statistics:")
        stats = self.au_emo_matrix.get_statistics()
        for key, value in stats.items():
            summary.append(f"  {key}: {value:.4f}")

        summary.append("\nMatrix Evolution:")
        for entry in self.training_stats['matrix_evolution'][-5:]:  # Last 5
            summary.append(f"  Task {entry['task_id']}, Epoch {entry['epoch']}: "
                          f"KL={entry['kl_from_prior']:.4f}")

        return "\n".join(summary)
