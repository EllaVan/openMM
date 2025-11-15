"""
Evaluation Metrics for Continual Learning

Implements various metrics to evaluate continual learning performance:
- Average Accuracy
- Forgetting Measure
- Forward Transfer
- Backward Transfer
- Learning Curve Area
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)
import matplotlib.pyplot as plt
from pathlib import Path


class ContinualLearningMetrics:
    """
    Metrics calculator for continual learning

    Tracks performance across tasks and computes various CL-specific metrics.

    Key metrics:
    ------------
    - Average Accuracy: Mean accuracy across all tasks
    - Forgetting: How much performance drops on previous tasks
    - Forward Transfer: How well knowledge transfers to new tasks
    - Backward Transfer: How new tasks affect old task performance

    Parameters:
    -----------
    num_tasks : int
        Total number of tasks
    num_classes_per_task : Dict[int, int]
        Number of classes for each task
    """

    def __init__(
        self,
        num_tasks: int,
        num_classes_per_task: Optional[Dict[int, int]] = None
    ):
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task or {}

        # Performance matrix: R[i, j] = accuracy on task j after training task i
        self.performance_matrix = np.zeros((num_tasks, num_tasks))

        # Track which tasks have been evaluated
        self.evaluated_tasks = set()

        # Store predictions and labels for detailed analysis
        self.predictions_history = {}  # {(task_trained, task_eval): predictions}
        self.labels_history = {}       # {(task_trained, task_eval): labels}

    def update(
        self,
        task_trained: int,
        task_eval: int,
        predictions: np.ndarray,
        labels: np.ndarray
    ):
        """
        Update metrics with evaluation results

        Args:
            task_trained: Task ID that was just trained
            task_eval: Task ID being evaluated
            predictions: Model predictions [num_samples]
            labels: True labels [num_samples]
        """
        # Compute accuracy
        accuracy = accuracy_score(labels, predictions)

        # Update performance matrix
        self.performance_matrix[task_trained, task_eval] = accuracy

        # Mark as evaluated
        self.evaluated_tasks.add((task_trained, task_eval))

        # Store for detailed analysis
        self.predictions_history[(task_trained, task_eval)] = predictions
        self.labels_history[(task_trained, task_eval)] = labels

    def average_accuracy(self, task_trained: int) -> float:
        """
        Average accuracy after training task i

        AA_i = 1/i * Σ_{j=0}^{i} R_{i,j}

        Args:
            task_trained: Task ID that was just trained

        Returns:
            average accuracy
        """
        if task_trained == 0:
            return self.performance_matrix[0, 0]

        # Average over tasks 0 to task_trained
        accuracies = self.performance_matrix[task_trained, :task_trained+1]
        return np.mean(accuracies)

    def final_average_accuracy(self) -> float:
        """
        Final average accuracy after training all tasks

        AA = 1/T * Σ_{j=0}^{T-1} R_{T-1,j}
        """
        return self.average_accuracy(self.num_tasks - 1)

    def forgetting_measure(self, task_eval: int, current_task: int) -> float:
        """
        Forgetting measure for a specific task

        F_j = max_{k ∈ {1,...,T-1}} R_{k,j} - R_{T,j}

        Measures how much performance dropped from peak to final.

        Args:
            task_eval: Task to measure forgetting for
            current_task: Current task being trained

        Returns:
            forgetting measure (0 = no forgetting, >0 = forgetting occurred)
        """
        if task_eval >= current_task:
            return 0.0  # Cannot forget tasks not yet seen

        # Find peak performance on this task
        peak_accuracy = np.max(self.performance_matrix[:current_task+1, task_eval])

        # Current performance
        current_accuracy = self.performance_matrix[current_task, task_eval]

        # Forgetting = peak - current
        return max(0.0, peak_accuracy - current_accuracy)

    def average_forgetting(self, current_task: int) -> float:
        """
        Average forgetting across all previous tasks

        F = 1/(T-1) * Σ_{j=0}^{T-2} F_j

        Args:
            current_task: Current task being trained

        Returns:
            average forgetting
        """
        if current_task == 0:
            return 0.0

        forgettings = [
            self.forgetting_measure(task_eval, current_task)
            for task_eval in range(current_task)
        ]

        return np.mean(forgettings)

    def forward_transfer(self, task_id: int) -> float:
        """
        Forward transfer for task i

        Measures how well knowledge from previous tasks helps with new task.

        FWT_i = R_{i-1,i} - R_{rand,i}

        where R_{i-1,i} is performance on task i before training it,
        and R_{rand,i} is random baseline.

        Args:
            task_id: Task to measure forward transfer for

        Returns:
            forward transfer (>0 = positive transfer, <0 = negative transfer)
        """
        if task_id == 0:
            return 0.0  # No forward transfer for first task

        # Performance on task_id before training it
        if task_id > 0:
            perf_before = self.performance_matrix[task_id-1, task_id]
        else:
            perf_before = 0.0

        # Random baseline (assume 1/num_classes)
        num_classes = self.num_classes_per_task.get(task_id, 2)
        random_baseline = 1.0 / num_classes

        return perf_before - random_baseline

    def average_forward_transfer(self) -> float:
        """
        Average forward transfer across all tasks

        FWT = 1/(T-1) * Σ_{i=1}^{T-1} FWT_i
        """
        if self.num_tasks == 1:
            return 0.0

        fwts = [
            self.forward_transfer(task_id)
            for task_id in range(1, self.num_tasks)
        ]

        return np.mean(fwts)

    def backward_transfer(self, task_eval: int, current_task: int) -> float:
        """
        Backward transfer for task j evaluated after training task i

        BWT_{i,j} = R_{i,j} - R_{j,j}

        Measures how training task i affects performance on previous task j.

        Args:
            task_eval: Previous task to evaluate
            current_task: Current task that was just trained

        Returns:
            backward transfer (>0 = positive, <0 = negative/forgetting)
        """
        if task_eval >= current_task:
            return 0.0

        # Performance after training current task
        perf_after = self.performance_matrix[current_task, task_eval]

        # Performance right after training task_eval
        perf_original = self.performance_matrix[task_eval, task_eval]

        return perf_after - perf_original

    def average_backward_transfer(self, current_task: int) -> float:
        """
        Average backward transfer after training current task

        BWT = 1/(T-1) * Σ_{j=0}^{T-2} BWT_{T-1,j}
        """
        if current_task == 0:
            return 0.0

        bwts = [
            self.backward_transfer(task_eval, current_task)
            for task_eval in range(current_task)
        ]

        return np.mean(bwts)

    def learning_curve_area(self, task_id: int) -> float:
        """
        Area under learning curve for a task

        Measures total performance over time for a task.

        Args:
            task_id: Task to compute area for

        Returns:
            area under learning curve
        """
        # Get all performances on this task
        performances = self.performance_matrix[:, task_id]

        # Compute area (trapezoidal rule)
        area = np.trapz(performances)

        return area

    def get_confusion_matrix(
        self,
        task_trained: int,
        task_eval: int
    ) -> Optional[np.ndarray]:
        """
        Get confusion matrix for specific evaluation

        Args:
            task_trained: Task that was trained
            task_eval: Task that was evaluated

        Returns:
            confusion matrix or None if not available
        """
        if (task_trained, task_eval) not in self.predictions_history:
            return None

        predictions = self.predictions_history[(task_trained, task_eval)]
        labels = self.labels_history[(task_trained, task_eval)]

        return confusion_matrix(labels, predictions)

    def get_per_class_metrics(
        self,
        task_trained: int,
        task_eval: int
    ) -> Optional[Dict]:
        """
        Get per-class precision, recall, F1

        Returns:
            dict with precision, recall, f1, support per class
        """
        if (task_trained, task_eval) not in self.predictions_history:
            return None

        predictions = self.predictions_history[(task_trained, task_eval)]
        labels = self.labels_history[(task_trained, task_eval)]

        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support
        }

    def plot_performance_matrix(
        self,
        save_path: Optional[str] = None,
        task_names: Optional[List[str]] = None
    ):
        """
        Plot the performance matrix as a heatmap

        Args:
            save_path: Path to save figure (optional)
            task_names: Names for tasks (optional)
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create heatmap
        im = ax.imshow(self.performance_matrix, cmap='RdYlGn', aspect='auto',
                      vmin=0, vmax=1)

        # Set ticks and labels
        if task_names is None:
            task_names = [f"Task {i}" for i in range(self.num_tasks)]

        ax.set_xticks(np.arange(self.num_tasks))
        ax.set_yticks(np.arange(self.num_tasks))
        ax.set_xticklabels(task_names)
        ax.set_yticklabels(task_names)

        # Rotate labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

        # Add values in cells
        for i in range(self.num_tasks):
            for j in range(self.num_tasks):
                if (i, j) in self.evaluated_tasks:
                    text = ax.text(j, i, f"{self.performance_matrix[i, j]:.3f}",
                                 ha="center", va="center", color="black")

        ax.set_xlabel("Task Evaluated")
        ax.set_ylabel("After Training Task")
        ax.set_title("Performance Matrix")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Accuracy", rotation=270, labelpad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance matrix saved to {save_path}")

        return fig

    def plot_learning_curves(
        self,
        save_path: Optional[str] = None,
        task_names: Optional[List[str]] = None
    ):
        """
        Plot learning curves for all tasks

        Args:
            save_path: Path to save figure (optional)
            task_names: Names for tasks (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))

        if task_names is None:
            task_names = [f"Task {i}" for i in range(self.num_tasks)]

        # Plot curve for each task
        for task_id in range(self.num_tasks):
            performances = self.performance_matrix[:, task_id]
            # Only plot up to where task has been evaluated
            valid_perf = [p for i, p in enumerate(performances)
                         if (i, task_id) in self.evaluated_tasks]

            if valid_perf:
                x = list(range(len(valid_perf)))
                ax.plot(x, valid_perf, marker='o', label=task_names[task_id])

        ax.set_xlabel("Training Stage")
        ax.set_ylabel("Accuracy")
        ax.set_title("Learning Curves")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Learning curves saved to {save_path}")

        return fig

    def get_summary(self) -> str:
        """Get a text summary of all metrics"""
        summary = []
        summary.append("\n" + "="*80)
        summary.append("CONTINUAL LEARNING METRICS SUMMARY")
        summary.append("="*80)

        # Final metrics
        final_task = self.num_tasks - 1
        summary.append(f"\nFinal Metrics (after {self.num_tasks} tasks):")
        summary.append(f"  Average Accuracy: {self.final_average_accuracy():.4f}")
        summary.append(f"  Average Forgetting: {self.average_forgetting(final_task):.4f}")
        summary.append(f"  Average Forward Transfer: {self.average_forward_transfer():.4f}")
        summary.append(f"  Average Backward Transfer: {self.average_backward_transfer(final_task):.4f}")

        # Per-task metrics
        summary.append(f"\nPer-Task Performance:")
        for task_id in range(self.num_tasks):
            if (final_task, task_id) in self.evaluated_tasks:
                acc = self.performance_matrix[final_task, task_id]
                forget = self.forgetting_measure(task_id, final_task)
                summary.append(f"  Task {task_id}: Accuracy = {acc:.4f}, Forgetting = {forget:.4f}")

        # Performance matrix
        summary.append(f"\nPerformance Matrix:")
        summary.append("  " + " ".join([f"T{i:2d}" for i in range(self.num_tasks)]))
        for i in range(self.num_tasks):
            row = f"T{i}: "
            for j in range(self.num_tasks):
                if (i, j) in self.evaluated_tasks:
                    row += f"{self.performance_matrix[i, j]:.2f} "
                else:
                    row += "  -  "
            summary.append(row)

        summary.append("="*80)

        return "\n".join(summary)

    def save_metrics(self, filepath: str):
        """Save metrics to file"""
        import json

        metrics = {
            'num_tasks': self.num_tasks,
            'performance_matrix': self.performance_matrix.tolist(),
            'final_average_accuracy': self.final_average_accuracy(),
            'average_forgetting': self.average_forgetting(self.num_tasks - 1),
            'average_forward_transfer': self.average_forward_transfer(),
            'average_backward_transfer': self.average_backward_transfer(self.num_tasks - 1),
            'per_task_performance': {
                f'task_{i}': {
                    'accuracy': self.performance_matrix[self.num_tasks-1, i],
                    'forgetting': self.forgetting_measure(i, self.num_tasks-1)
                }
                for i in range(self.num_tasks)
            }
        }

        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {filepath}")


if __name__ == "__main__":
    print("Testing Continual Learning Metrics...")

    # Simulate 3 tasks
    metrics = ContinualLearningMetrics(num_tasks=3)

    # Simulate some evaluations
    np.random.seed(42)

    for task_trained in range(3):
        for task_eval in range(task_trained + 1):
            # Simulate predictions
            num_samples = 100
            labels = np.random.randint(0, 3, num_samples)
            # Accuracy decreases for old tasks (simulating forgetting)
            accuracy_target = 0.9 - 0.1 * (task_trained - task_eval)
            predictions = labels.copy()
            wrong_indices = np.random.choice(
                num_samples,
                size=int(num_samples * (1 - accuracy_target)),
                replace=False
            )
            predictions[wrong_indices] = (predictions[wrong_indices] + 1) % 3

            metrics.update(task_trained, task_eval, predictions, labels)

    # Print summary
    print(metrics.get_summary())

    # Test metrics
    print(f"\nFinal AA: {metrics.final_average_accuracy():.4f}")
    print(f"Avg Forgetting: {metrics.average_forgetting(2):.4f}")

    print("\n✓ Metrics module ready!")
