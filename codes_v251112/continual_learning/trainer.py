"""
Continual Learning Trainer for Multi-domain Zero-shot Emotion Recognition

This module implements the main training loop for continual learning across
multiple domains with seen and unseen emotion classes.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import time
from tqdm import tqdm
import numpy as np

from continual_learning.au_emotion_network import AUEmotionNetwork
from continual_learning.consistency_checker import (
    MultimodalConsistencyChecker,
    ConsistencyStrategy
)
from continual_learning.ewc import EWC, OnlineEWC


class ContinualLearningTrainer:
    """
    Continual Learning Trainer

    Manages training across multiple tasks/domains with seen and unseen classes.

    Training procedure for each task:
    1. Train on seen class samples with labels
    2. Update AU-EMO matrix using seen class labels
    3. Predict unseen class samples using AU-EMO matrix
    4. Filter reliable predictions using consistency checker
    5. Update AU-EMO matrix using consistent unseen predictions
    6. Apply EWC penalty to prevent forgetting previous tasks
    7. Evaluate on all seen/unseen classes from current and previous tasks

    Parameters:
    -----------
    model : AUEmotionNetwork
        The AU-based emotion recognition network
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters
    device : str
        Device to use ('cuda' or 'cpu')
    use_ewc : bool
        Whether to use EWC for防遗忘
    ewc_lambda : float
        EWC regularization weight
    ewc_type : str
        Type of EWC ('standard', 'online', or 'selective')
    consistency_strategy : ConsistencyStrategy
        Strategy for checking multimodal consistency
    min_confidence : float
        Minimum confidence for unseen class updates
    seen_update_weight : float
        Weight for seen class AU-EMO updates
    unseen_update_weight : float
        Weight for unseen class AU-EMO updates
    au_emo_regularization : float
        Regularization strength towards AU-EMO prior
    save_dir : str
        Directory to save checkpoints
    """

    def __init__(
        self,
        model: AUEmotionNetwork,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        use_ewc: bool = True,
        ewc_lambda: float = 1000.0,
        ewc_type: str = 'online',
        consistency_strategy: ConsistencyStrategy = ConsistencyStrategy.MAJORITY,
        min_confidence: float = 0.8,
        seen_update_weight: float = 10.0,
        unseen_update_weight: float = 1.0,
        au_emo_regularization: float = 0.01,
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device

        # EWC for anti-forgetting
        self.use_ewc = use_ewc
        if use_ewc:
            if ewc_type == 'online':
                self.ewc = OnlineEWC(model, ewc_lambda=ewc_lambda, device=device)
            elif ewc_type == 'selective':
                from continual_learning.ewc import SelectiveEWC
                self.ewc = SelectiveEWC(model, ewc_lambda=ewc_lambda, device=device)
            else:
                self.ewc = EWC(model, ewc_lambda=ewc_lambda, device=device)
        else:
            self.ewc = None

        # Consistency checker for unseen classes
        self.consistency_checker = MultimodalConsistencyChecker(
            model=model,
            strategy=consistency_strategy,
            min_confidence=min_confidence
        )

        # Training hyperparameters
        self.seen_update_weight = seen_update_weight
        self.unseen_update_weight = unseen_update_weight
        self.au_emo_regularization = au_emo_regularization

        # Save directory
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training statistics
        self.task_history = []
        self.current_task = 0

    def train_task(
        self,
        task_id: int,
        task_name: str,
        seen_loader: DataLoader,
        unseen_loader: Optional[DataLoader],
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        log_interval: int = 10,
        evaluate_interval: int = 1
    ) -> Dict:
        """
        Train on a single task

        Args:
            task_id: Task identifier
            task_name: Task name (e.g., "MOSEI_happy_sad")
            seen_loader: DataLoader for seen class samples
            unseen_loader: DataLoader for unseen class samples (optional)
            val_loader: Validation DataLoader (optional)
            num_epochs: Number of training epochs
            log_interval: How often to log progress (batches)
            evaluate_interval: How often to evaluate (epochs)

        Returns:
            dict: Training statistics
        """
        print(f"\n{'='*80}")
        print(f"Training Task {task_id}: {task_name}")
        print(f"{'='*80}")

        self.current_task = task_id
        task_stats = {
            'task_id': task_id,
            'task_name': task_name,
            'epochs': num_epochs,
            'train_history': []
        }

        # Training loop
        for epoch in range(num_epochs):
            epoch_start = time.time()

            # ===== Phase 1: Train on seen classes =====
            seen_stats = self._train_seen_epoch(
                seen_loader,
                epoch,
                log_interval
            )

            # ===== Phase 2: Train on unseen classes (if available) =====
            if unseen_loader is not None:
                unseen_stats = self._train_unseen_epoch(
                    unseen_loader,
                    epoch,
                    log_interval
                )
            else:
                unseen_stats = {}

            # ===== Regularize AU-EMO matrix =====
            if epoch % 5 == 0:  # Every 5 epochs
                kl_div = self.model.regularize_au_emo_matrix(
                    self.au_emo_regularization
                )
                print(f"AU-EMO regularization: KL div = {kl_div:.4f}")

            # ===== Evaluate =====
            if val_loader is not None and epoch % evaluate_interval == 0:
                val_stats = self.evaluate(val_loader, phase='validation')
            else:
                val_stats = {}

            # Record epoch statistics
            epoch_time = time.time() - epoch_start
            epoch_stats = {
                'epoch': epoch,
                'seen': seen_stats,
                'unseen': unseen_stats,
                'validation': val_stats,
                'epoch_time': epoch_time
            }
            task_stats['train_history'].append(epoch_stats)

            print(f"\nEpoch {epoch} completed in {epoch_time:.2f}s")
            print(f"  Seen loss: {seen_stats['avg_loss']:.4f}, "
                  f"acc: {seen_stats['accuracy']:.4f}")
            if unseen_stats:
                print(f"  Unseen consistent: {unseen_stats['consistent_samples']}, "
                      f"rate: {unseen_stats['consistency_rate']:.4f}")

        # ===== Consolidate task (EWC) =====
        if self.use_ewc:
            print("\nConsolidating task for EWC...")
            self.ewc.consolidate(seen_loader, num_samples=1000)

        # Save task checkpoint
        self._save_task_checkpoint(task_id, task_name)

        # Record task history
        self.task_history.append(task_stats)

        return task_stats

    def _train_seen_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int
    ) -> Dict:
        """Train one epoch on seen class samples"""
        self.model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        au_emo_updates = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Seen]")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            batch = self._move_batch_to_device(batch)

            # Forward pass
            output = self.model(
                batch['text_features'],
                batch['audio_features'],
                batch['video_features'],
                batch['masks'],
                batch['labels']
            )

            # Compute loss
            loss = output['loss']

            # Add EWC penalty
            if self.use_ewc:
                ewc_penalty = self.ewc.penalty()
                loss = loss + ewc_penalty

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update AU-EMO matrix with seen samples
            with torch.no_grad():
                update_stats = self.model.update_au_emo_matrix(
                    au_probs=output['au_probs'],
                    emo_labels=batch['labels'],
                    is_seen=True,
                    seen_weight=self.seen_update_weight
                )
                au_emo_updates += update_stats['updated_samples']

            # Statistics
            batch_size = batch['labels'].size(0)
            total_loss += loss.item() * batch_size
            predictions = torch.argmax(output['emo_from_au'], dim=1)
            total_correct += (predictions == batch['labels']).sum().item()
            total_samples += batch_size

            # Update progress bar
            if batch_idx % log_interval == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{total_correct/total_samples:.4f}"
                })

        return {
            'avg_loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'total_samples': total_samples,
            'au_emo_updates': au_emo_updates
        }

    def _train_unseen_epoch(
        self,
        dataloader: DataLoader,
        epoch: int,
        log_interval: int
    ) -> Dict:
        """Train one epoch on unseen class samples"""
        self.model.eval()  # Use eval mode for unseen (no gradient updates)

        total_consistent = 0
        total_samples = 0
        au_emo_updates = 0
        confidence_scores = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Unseen]")

        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                # Move to device
                batch = self._move_batch_to_device(batch)

                # Check multimodal consistency
                consistency_result = self.consistency_checker.check_consistency(
                    batch['text_features'],
                    batch['audio_features'],
                    batch['video_features'],
                    batch['masks']
                )

                is_consistent = consistency_result['is_consistent']
                consensus_label = consistency_result['consensus_label']
                confidence = consistency_result['confidence']

                # Update AU-EMO matrix only for consistent samples
                if is_consistent.any():
                    # Get AU predictions for consistent samples
                    output = self.model(
                        batch['text_features'][is_consistent],
                        batch['audio_features'][is_consistent],
                        batch['video_features'][is_consistent],
                        batch['masks'][is_consistent] if batch['masks'] is not None else None
                    )

                    update_stats = self.model.update_au_emo_matrix(
                        au_probs=output['au_probs'],
                        emo_labels=consensus_label[is_consistent],
                        is_seen=False,
                        confidence=confidence[is_consistent],
                        unseen_weight=self.unseen_update_weight
                    )
                    au_emo_updates += update_stats['updated_samples']

                # Statistics
                batch_size = batch['text_features'].size(0)
                total_consistent += is_consistent.sum().item()
                total_samples += batch_size
                confidence_scores.extend(confidence[is_consistent].cpu().tolist())

                # Update progress bar
                if batch_idx % log_interval == 0:
                    pbar.set_postfix({
                        'consistent': f"{total_consistent}/{total_samples}",
                        'rate': f"{total_consistent/total_samples:.4f}"
                    })

        consistency_rate = total_consistent / total_samples if total_samples > 0 else 0.0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0

        return {
            'consistent_samples': total_consistent,
            'total_samples': total_samples,
            'consistency_rate': consistency_rate,
            'avg_confidence': avg_confidence,
            'au_emo_updates': au_emo_updates
        }

    def evaluate(
        self,
        dataloader: DataLoader,
        phase: str = 'test'
    ) -> Dict:
        """
        Evaluate model on a dataset

        Args:
            dataloader: DataLoader for evaluation
            phase: Evaluation phase name ('validation', 'test', etc.)

        Returns:
            dict: Evaluation metrics
        """
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating [{phase}]"):
                batch = self._move_batch_to_device(batch)

                # Predict using AU path
                predictions, confidence = self.model.predict_from_au(
                    batch['text_features'],
                    batch['audio_features'],
                    batch['video_features'],
                    batch['masks']
                )

                all_predictions.append(predictions.cpu())
                if 'labels' in batch:
                    all_labels.append(batch['labels'].cpu())
                all_confidences.append(confidence.cpu())

        # Concatenate results
        all_predictions = torch.cat(all_predictions)
        all_confidences = torch.cat(all_confidences)

        results = {
            'predictions': all_predictions.numpy(),
            'confidences': all_confidences.numpy(),
            'avg_confidence': all_confidences.mean().item()
        }

        # Compute accuracy if labels available
        if all_labels:
            all_labels = torch.cat(all_labels)
            accuracy = (all_predictions == all_labels).float().mean().item()
            results['labels'] = all_labels.numpy()
            results['accuracy'] = accuracy
            print(f"{phase.capitalize()} Accuracy: {accuracy:.4f}")

        return results

    def _move_batch_to_device(self, batch: Dict) -> Dict:
        """Move batch to device"""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    def _save_task_checkpoint(self, task_id: int, task_name: str):
        """Save checkpoint after completing a task"""
        checkpoint_path = self.save_dir / f"task_{task_id}_{task_name}.pt"

        checkpoint = {
            'task_id': task_id,
            'task_name': task_name,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'au_emo_statistics': self.model.get_au_emo_statistics(),
            'task_history': self.task_history
        }

        if self.use_ewc:
            checkpoint['ewc_state'] = {
                'consolidated_tasks': self.ewc.consolidated_tasks,
                'ewc_lambda': self.ewc.ewc_lambda
            }

        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # Also save AU-EMO matrix separately
        au_emo_path = self.save_dir / f"au_emo_matrix_task_{task_id}.npz"
        self.model.save_au_emo_matrix(str(au_emo_path))

    def save_final_model(self, filename: str = 'final_model.pt'):
        """Save final model after all tasks"""
        save_path = self.save_dir / filename

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'au_emo_statistics': self.model.get_au_emo_statistics(),
            'task_history': self.task_history,
            'total_tasks': self.current_task + 1
        }

        if self.use_ewc:
            ewc_path = self.save_dir / 'ewc_final.pt'
            self.ewc.save(str(ewc_path))

        torch.save(checkpoint, save_path)
        print(f"\nFinal model saved: {save_path}")

        # Save AU-EMO matrix
        au_emo_path = self.save_dir / 'au_emo_matrix_final.npz'
        self.model.save_au_emo_matrix(str(au_emo_path))

        # Save training history as JSON
        history_path = self.save_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.task_history, f, indent=2, default=str)
        print(f"Training history saved: {history_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'task_history' in checkpoint:
            self.task_history = checkpoint['task_history']
            self.current_task = checkpoint['task_id']

        print(f"Checkpoint loaded: {checkpoint_path}")

    def get_training_summary(self) -> str:
        """Get a text summary of training progress"""
        summary = []
        summary.append("\n" + "="*80)
        summary.append("CONTINUAL LEARNING TRAINING SUMMARY")
        summary.append("="*80)

        summary.append(f"\nTotal tasks completed: {len(self.task_history)}")

        for task_stat in self.task_history:
            summary.append(f"\nTask {task_stat['task_id']}: {task_stat['task_name']}")
            summary.append(f"  Epochs: {task_stat['epochs']}")

            if task_stat['train_history']:
                last_epoch = task_stat['train_history'][-1]
                summary.append(f"  Final seen accuracy: {last_epoch['seen']['accuracy']:.4f}")

                if last_epoch['unseen']:
                    summary.append(f"  Unseen consistency rate: {last_epoch['unseen']['consistency_rate']:.4f}")

        # AU-EMO statistics
        au_stats = self.model.get_au_emo_statistics()
        summary.append(f"\nAU-EMO Matrix Statistics:")
        summary.append(f"  Total updates: {au_stats['total_updates']}")
        summary.append(f"  Seen updates: {au_stats['seen_updates']}")
        summary.append(f"  Unseen updates: {au_stats['unseen_updates']}")
        summary.append(f"  Matrix entropy: {au_stats['matrix_entropy']:.4f}")

        # EWC statistics
        if self.use_ewc:
            summary.append(f"\nEWC Statistics:")
            summary.append(f"  Consolidated tasks: {self.ewc.consolidated_tasks}")

        summary.append("\n" + "="*80)

        return "\n".join(summary)


if __name__ == "__main__":
    print("Continual Learning Trainer module ready!")
    print("\nExample usage:")
    print("""
    # Create model
    model = AUEmotionNetwork(...)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Create trainer
    trainer = ContinualLearningTrainer(
        model=model,
        optimizer=optimizer,
        use_ewc=True,
        consistency_strategy=ConsistencyStrategy.MAJORITY
    )

    # Train on tasks
    for task_id, (seen_loader, unseen_loader) in enumerate(task_loaders):
        trainer.train_task(
            task_id=task_id,
            task_name=f"Task_{task_id}",
            seen_loader=seen_loader,
            unseen_loader=unseen_loader,
            num_epochs=10
        )

    # Save final model
    trainer.save_final_model()

    # Print summary
    print(trainer.get_training_summary())
    """)
