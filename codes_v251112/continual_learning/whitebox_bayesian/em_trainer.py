"""
EM Algorithm Trainer for Whitebox Beta-Bernoulli AU-EMO Matrix

Implements Expectation-Maximization (EM) algorithm to jointly optimize:
- AU predictor network (E-step)
- AU-EMO probability matrix (M-step)

This avoids circular dependency and instability from simultaneous updates.
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

from beta_bernoulli_matrix import BetaBernoulliAUEMOMatrix
from continual_learning.au_emotion_network import AUEmotionNetwork
from continual_learning.consistency_checker import MultimodalConsistencyChecker, ConsistencyStrategy
from continual_learning.ewc import OnlineEWC


class EMTrainerWhitebox:
    """
    EM Algorithm Trainer for Whitebox Continual Learning

    Training Flow:
    --------------
    Task 0:
        1. Warmup: Train AU predictor for few epochs
        2. Seen EM: Alternate E-step and M-step
        3. Unseen: Consistency checking + matrix update

    Task 1 to T:
        1. Seen EM: E-step and M-step with EWC
        2. Unseen: Consistency checking + matrix update
        3. Consolidate EWC

    E-step (Expectation):
        - Freeze Beta-Bernoulli matrix
        - Optimize AU predictor network
        - Loss: emotion classification via AU-EMO matrix

    M-step (Maximization):
        - Freeze AU predictor
        - Collect AU predictions
        - Update Beta parameters using Bayesian update

    Parameters:
    -----------
    model : AUEmotionNetwork
        Multimodal network with AU predictor
    au_emo_matrix : BetaBernoulliAUEMOMatrix
        Beta-Bernoulli AU-EMO matrix
    optimizer : torch.optim.Optimizer
        Optimizer for network parameters
    device : str
        Device for training
    use_ewc : bool
        Whether to use EWC for anti-forgetting
    ewc_lambda : float
        EWC regularization strength
    consistency_strategy : ConsistencyStrategy
        Strategy for multimodal consistency checking
    min_confidence : float
        Minimum confidence for unseen updates
    seen_update_weight : float
        Weight for seen class matrix updates
    unseen_update_weight : float
        Weight for unseen class matrix updates
    save_dir : str
        Directory to save checkpoints
    """

    def __init__(
        self,
        model: AUEmotionNetwork,
        au_emo_matrix: BetaBernoulliAUEMOMatrix,
        optimizer: optim.Optimizer,
        device: str = 'cuda',
        use_ewc: bool = True,
        ewc_lambda: float = 1000.0,
        consistency_strategy: ConsistencyStrategy = ConsistencyStrategy.MAJORITY,
        min_confidence: float = 0.8,
        seen_update_weight: float = 1.0,
        unseen_update_weight: float = 0.8,
        au_emo_regularization: float = 0.01,
        save_dir: str = './checkpoints/whitebox'
    ):
        self.model = model.to(device)
        self.au_emo_matrix = au_emo_matrix
        self.optimizer = optimizer
        self.device = device

        # Continual learning settings
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda
        self.ewc = OnlineEWC(model, device=device) if use_ewc else None

        # Consistency checking
        self.consistency_checker = MultimodalConsistencyChecker(
            model=model,
            strategy=consistency_strategy,
            device=device
        )
        self.min_confidence = min_confidence

        # AU-EMO matrix update settings
        self.seen_update_weight = seen_update_weight
        self.unseen_update_weight = unseen_update_weight
        self.au_emo_regularization = au_emo_regularization

        # Directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.training_stats = {
            'tasks': [],
            'em_iterations': [],
            'matrix_updates': []
        }

    def train_task(
        self,
        task_id: int,
        task_name: str,
        seen_loader: torch.utils.data.DataLoader,
        unseen_loader: Optional[torch.utils.data.DataLoader] = None,
        num_epochs: int = 10,
        num_em_iterations: int = 3,
        log_interval: int = 10
    ) -> Dict:
        """
        Train on a single task using EM algorithm

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
        num_em_iterations : int
            Number of EM iterations per epoch
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

        # Phase 2: Seen EM Training
        print(f"\n[Task {task_id}] Phase 2: Seen EM Training")
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")

            # Run EM iterations
            for em_iter in range(num_em_iterations):
                # E-step: Optimize AU predictor
                e_stats = self._e_step(seen_loader, log_interval=log_interval)

                # M-step: Update Beta-Bernoulli matrix
                m_stats = self._m_step(seen_loader, is_seen=True)

                print(f"  EM Iter {em_iter+1}/{num_em_iterations}: "
                      f"E-step loss={e_stats['avg_loss']:.4f}, "
                      f"M-step updated={m_stats['updated_samples']}")

            # Periodic regularization towards prior
            if (epoch + 1) % 5 == 0:
                kl_div = self.au_emo_matrix.regularize_to_prior(
                    strength=self.au_emo_regularization
                )
                print(f"  Regularized to prior (KL={kl_div:.4f})")

            # Evaluate
            eval_stats = self._evaluate_seen(seen_loader)
            task_stats['epochs'].append(epoch)
            task_stats['seen_metrics'].append(eval_stats)

            print(f"  Seen evaluation: acc={eval_stats['accuracy']:.4f}")

        # Phase 3: Unseen Training (if available)
        if unseen_loader is not None:
            print(f"\n[Task {task_id}] Phase 3: Unseen Training")
            unseen_stats = self._unseen_update_validated(
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
        Warmup phase: Train AU predictor with frozen matrix

        Goal: Initialize AU predictor to produce reasonable AU predictions
        """
        print("Warmup: Training AU predictor with frozen matrix...")

        self.model.train()

        for epoch in range(num_epochs):
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

                # Loss: emotion classification via AU path
                loss = F.cross_entropy(outputs['emo_from_au'], labels)

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

    def _e_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        log_interval: int = 10
    ) -> Dict:
        """
        E-step: Optimize AU predictor with frozen Beta-Bernoulli matrix

        Returns:
        --------
        stats : dict
            E-step statistics
        """
        self.model.train()

        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            # Move to device
            text = batch['text'].to(self.device)
            audio = batch['audio'].to(self.device)
            video = batch['video'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            outputs = self.model(text, audio, video, batch.get('masks'))

            # E-step loss: emotion classification via AU-EMO matrix
            loss_emo = F.cross_entropy(outputs['emo_from_au'], labels)

            # Optional: Direct path auxiliary loss
            loss_direct = F.cross_entropy(outputs['emo_direct'], labels)

            # Combined loss
            loss = loss_emo + 0.1 * loss_direct

            # Add EWC penalty if applicable
            if self.use_ewc and self.ewc.is_consolidated:
                loss += self.ewc_lambda * self.ewc.penalty()

            # Backward and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return {
            'avg_loss': total_loss / num_batches,
            'num_batches': num_batches
        }

    def _m_step(
        self,
        dataloader: torch.utils.data.DataLoader,
        is_seen: bool = True,
        confidence_threshold: Optional[float] = None
    ) -> Dict:
        """
        M-step: Update Beta-Bernoulli matrix with frozen AU predictor

        Returns:
        --------
        stats : dict
            M-step statistics
        """
        self.model.eval()

        total_updated = 0
        total_skipped = 0
        all_confidences = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                # Get AU predictions
                outputs = self.model(text, audio, video, batch.get('masks'))
                au_probs = outputs['au_probs']

                # Get confidence (for unseen)
                if not is_seen:
                    emo_probs = F.softmax(outputs['emo_from_au'], dim=1)
                    confidence, _ = emo_probs.max(dim=1)
                else:
                    confidence = None

                # Update Beta-Bernoulli matrix
                update_stats = self.au_emo_matrix.update_from_labels(
                    au_probs=au_probs,
                    emo_labels=labels,
                    is_seen=is_seen,
                    confidence=confidence,
                    seen_weight=self.seen_update_weight,
                    unseen_weight=self.unseen_update_weight,
                    min_confidence=confidence_threshold or self.min_confidence
                )

                total_updated += update_stats['updated_samples']
                total_skipped += update_stats['skipped_samples']
                if 'avg_confidence' in update_stats:
                    all_confidences.append(update_stats['avg_confidence'])

        return {
            'updated_samples': total_updated,
            'skipped_samples': total_skipped,
            'avg_confidence': np.mean(all_confidences) if all_confidences else 1.0
        }

    def _unseen_update_validated(
        self,
        unseen_loader: torch.utils.data.DataLoader,
        num_epochs: int = 5,
        log_interval: int = 10,
        validation_interval: int = 2
    ) -> Dict:
        """
        Unseen update with validation and rollback

        Strategy:
        1. Use consistency checker to get pseudo-labels
        2. Update matrix with high-confidence predictions
        3. Periodically validate on held-out set
        4. Rollback if performance degrades

        Returns:
        --------
        stats : dict
            Unseen training statistics
        """
        print("Unseen training with validation...")

        # Save initial state for potential rollback
        best_matrix_state = {
            'alpha': self.au_emo_matrix.alpha.clone(),
            'beta': self.au_emo_matrix.beta.clone()
        }
        best_performance = 0.0

        stats_per_epoch = []

        for epoch in range(num_epochs):
            print(f"\n  Unseen Epoch {epoch+1}/{num_epochs}")

            # Collect pseudo-labels from consistency checking
            pseudo_labels_data = self._collect_pseudo_labels(unseen_loader)

            if pseudo_labels_data['num_consistent'] == 0:
                print(f"    No consistent samples found, skipping epoch")
                continue

            print(f"    Consistent samples: {pseudo_labels_data['num_consistent']}/{pseudo_labels_data['total_samples']} "
                  f"({100*pseudo_labels_data['consistency_rate']:.1f}%)")

            # Update matrix with pseudo-labels
            update_stats = self._update_with_pseudo_labels(pseudo_labels_data)

            print(f"    Matrix updated with {update_stats['updated_samples']} samples")

            # Validation
            if (epoch + 1) % validation_interval == 0:
                # Evaluate consistency rate as proxy for quality
                current_performance = pseudo_labels_data['consistency_rate']

                if current_performance > best_performance:
                    best_performance = current_performance
                    # Save best state
                    best_matrix_state = {
                        'alpha': self.au_emo_matrix.alpha.clone(),
                        'beta': self.au_emo_matrix.beta.clone()
                    }
                    print(f"    New best consistency rate: {best_performance:.4f}")
                else:
                    print(f"    Performance did not improve, rolling back...")
                    # Rollback to best state
                    self.au_emo_matrix.alpha.copy_(best_matrix_state['alpha'])
                    self.au_emo_matrix.beta.copy_(best_matrix_state['beta'])

            stats_per_epoch.append({
                'epoch': epoch,
                'consistency_rate': pseudo_labels_data['consistency_rate'],
                'updated_samples': update_stats['updated_samples']
            })

        return {
            'best_consistency_rate': best_performance,
            'epochs': stats_per_epoch
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
            Contains AU predictions, pseudo-labels, and confidences
        """
        self.model.eval()

        all_au_probs = []
        all_pseudo_labels = []
        all_confidences = []
        all_is_consistent = []

        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                text = batch['text'].to(self.device)
                audio = batch['audio'].to(self.device)
                video = batch['video'].to(self.device)

                # Get AU predictions
                outputs = self.model(text, audio, video, batch.get('masks'))
                au_probs = outputs['au_probs']

                # Consistency checking
                results = self.consistency_checker.check_consistency(
                    text, audio, video, batch.get('masks')
                )

                all_au_probs.append(au_probs)
                all_pseudo_labels.append(results['consensus_label'])
                all_confidences.append(results['confidence'])
                all_is_consistent.append(results['is_consistent'])

        # Concatenate
        all_au_probs = torch.cat(all_au_probs, dim=0)
        all_pseudo_labels = torch.cat(all_pseudo_labels, dim=0)
        all_confidences = torch.cat(all_confidences, dim=0)
        all_is_consistent = torch.cat(all_is_consistent, dim=0)

        # Filter to consistent samples only
        consistent_mask = all_is_consistent

        return {
            'au_probs': all_au_probs[consistent_mask],
            'pseudo_labels': all_pseudo_labels[consistent_mask],
            'confidences': all_confidences[consistent_mask],
            'num_consistent': consistent_mask.sum().item(),
            'total_samples': len(all_is_consistent),
            'consistency_rate': consistent_mask.float().mean().item()
        }

    def _update_with_pseudo_labels(self, pseudo_labels_data: Dict) -> Dict:
        """Update matrix with collected pseudo-labels"""
        if pseudo_labels_data['num_consistent'] == 0:
            return {'updated_samples': 0}

        return self.au_emo_matrix.update_from_labels(
            au_probs=pseudo_labels_data['au_probs'],
            emo_labels=pseudo_labels_data['pseudo_labels'],
            is_seen=False,
            confidence=pseudo_labels_data['confidences'],
            seen_weight=self.seen_update_weight,
            unseen_weight=self.unseen_update_weight,
            min_confidence=self.min_confidence
        )

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
        summary.append("WHITEBOX TRAINING SUMMARY")
        summary.append("="*80)

        summary.append(f"\nTotal tasks trained: {len(self.training_stats['tasks'])}")

        summary.append("\nAU-EMO Matrix Statistics:")
        stats = self.au_emo_matrix.get_statistics()
        for key, value in stats.items():
            summary.append(f"  {key}: {value:.4f}")

        return "\n".join(summary)
