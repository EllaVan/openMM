"""
Elastic Weight Consolidation (EWC) for Continual Learning

EWC prevents catastrophic forgetting by constraining important parameters
to stay close to their values after training on previous tasks.

Reference:
Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks"
PNAS 2017
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List
from torch.utils.data import DataLoader
import copy


class EWC:
    """
    Elastic Weight Consolidation

    Computes Fisher Information Matrix to identify important parameters
    and adds a regularization penalty to prevent them from changing too much.

    The EWC penalty is:
        L_EWC = λ/2 * Σ_i F_i * (θ_i - θ*_i)^2

    where:
        - F_i is the Fisher information for parameter i
        - θ_i is the current parameter value
        - θ*_i is the optimal parameter value from previous task
        - λ is the importance weight

    Parameters:
    -----------
    model : nn.Module
        The neural network model
    ewc_lambda : float
        Importance weight for EWC penalty (default: 1000)
    device : str
        Device to store tensors ('cuda' or 'cpu')
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        device: str = 'cuda'
    ):
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.device = device

        # Store Fisher information and optimal parameters for each task
        self.fisher_dict = {}  # {param_name: fisher_info}
        self.optimal_params = {}  # {param_name: optimal_value}

        # Track which tasks have been consolidated
        self.consolidated_tasks = 0

    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
        empirical: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher Information Matrix

        The Fisher information measures the sensitivity of the loss
        to each parameter. We approximate it using the gradient of
        the log-likelihood on the training data.

        Args:
            dataloader: DataLoader with training data from current task
            num_samples: Maximum number of samples to use (None = use all)
            empirical: If True, use empirical Fisher (based on sampled labels)
                      If False, use true Fisher (based on predicted labels)

        Returns:
            fisher_dict: Dictionary mapping parameter names to Fisher values
        """
        fisher_dict = {}

        # Initialize Fisher matrices
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_dict[name] = torch.zeros_like(param, device=self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Accumulate Fisher information
        sample_count = 0

        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()}
                has_labels = 'labels' in batch
            else:
                batch = tuple(t.to(self.device) for t in batch)
                has_labels = True

            # Forward pass
            self.model.zero_grad()

            try:
                if isinstance(batch, dict):
                    # Get predictions
                    output = self.model(
                        batch['text_features'],
                        batch['audio_features'],
                        batch['video_features'],
                        batch.get('masks', None)
                    )

                    logits = output['emo_from_au']  # Use AU path for consistency

                    # Use empirical or true Fisher
                    if empirical and has_labels:
                        # Empirical: use true labels
                        labels = batch['labels']
                    else:
                        # True Fisher: use predicted labels
                        labels = torch.argmax(logits, dim=1)

                    # Compute loss
                    loss = F.cross_entropy(logits, labels)

                else:
                    # Assume batch is (features, labels)
                    features, labels = batch
                    logits = self.model(features)

                    if not empirical:
                        labels = torch.argmax(logits, dim=1)

                    loss = F.cross_entropy(logits, labels)

                # Backward pass to get gradients
                loss.backward()

                # Accumulate squared gradients (Fisher information)
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        fisher_dict[name] += param.grad.pow(2).detach()

                sample_count += len(labels)

                # Check if we've processed enough samples
                if num_samples is not None and sample_count >= num_samples:
                    break

            except Exception as e:
                print(f"Warning: Error computing Fisher for batch: {e}")
                continue

        # Average Fisher information
        for name in fisher_dict:
            fisher_dict[name] /= sample_count

        print(f"Fisher information computed using {sample_count} samples")

        return fisher_dict

    def consolidate(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
        online: bool = False
    ):
        """
        Consolidate current task

        Computes Fisher information and saves optimal parameters.

        Args:
            dataloader: DataLoader with training data
            num_samples: Maximum number of samples to use
            online: If True, accumulate Fisher (for online EWC)
                   If False, replace Fisher (for standard EWC)
        """
        print(f"\nConsolidating task {self.consolidated_tasks + 1}...")

        # Compute Fisher for current task
        current_fisher = self.compute_fisher(dataloader, num_samples)

        # Save optimal parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if online and name in self.fisher_dict:
                    # Online EWC: accumulate Fisher
                    self.fisher_dict[name] += current_fisher[name]
                else:
                    # Standard EWC: replace Fisher
                    self.fisher_dict[name] = current_fisher[name]

                # Always update optimal parameters
                self.optimal_params[name] = param.data.clone().detach()

        self.consolidated_tasks += 1

        # Print statistics
        total_fisher = sum(f.sum().item() for f in self.fisher_dict.values())
        print(f"  Total Fisher information: {total_fisher:.4f}")
        print(f"  Consolidated tasks: {self.consolidated_tasks}")

    def penalty(self) -> torch.Tensor:
        """
        Compute EWC regularization penalty

        Returns:
            penalty: EWC loss term
        """
        if self.consolidated_tasks == 0:
            # No previous tasks, no penalty
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                # EWC penalty: F * (θ - θ*)^2
                fisher = self.fisher_dict[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal).pow(2)).sum()

        # Scale by lambda and divide by 2
        return (self.ewc_lambda / 2.0) * loss

    def get_importance_map(self) -> Dict[str, torch.Tensor]:
        """
        Get parameter importance map

        Returns normalized Fisher information for each parameter.
        """
        if not self.fisher_dict:
            return {}

        # Find max Fisher value for normalization
        max_fisher = max(f.max().item() for f in self.fisher_dict.values())

        importance_map = {}
        for name, fisher in self.fisher_dict.items():
            importance_map[name] = fisher / (max_fisher + 1e-10)

        return importance_map

    def save(self, filepath: str):
        """Save EWC state"""
        state = {
            'fisher_dict': {name: fisher.cpu() for name, fisher in self.fisher_dict.items()},
            'optimal_params': {name: param.cpu() for name, param in self.optimal_params.items()},
            'consolidated_tasks': self.consolidated_tasks,
            'ewc_lambda': self.ewc_lambda
        }
        torch.save(state, filepath)
        print(f"EWC state saved to {filepath}")

    def load(self, filepath: str):
        """Load EWC state"""
        state = torch.load(filepath, map_location=self.device)

        self.fisher_dict = {name: fisher.to(self.device)
                           for name, fisher in state['fisher_dict'].items()}
        self.optimal_params = {name: param.to(self.device)
                              for name, param in state['optimal_params'].items()}
        self.consolidated_tasks = state['consolidated_tasks']
        self.ewc_lambda = state['ewc_lambda']

        print(f"EWC state loaded from {filepath}")
        print(f"  Consolidated tasks: {self.consolidated_tasks}")

    def reset(self):
        """Reset EWC (forget all previous tasks)"""
        self.fisher_dict = {}
        self.optimal_params = {}
        self.consolidated_tasks = 0
        print("EWC reset: all previous tasks forgotten")


class OnlineEWC(EWC):
    """
    Online EWC

    Extends EWC for online/streaming scenarios where we continually
    update Fisher information as new tasks arrive.

    This is more memory-efficient than storing separate Fisher matrices
    for each task.

    Parameters:
    -----------
    model : nn.Module
        The neural network model
    ewc_lambda : float
        Importance weight
    gamma : float
        Decay factor for online Fisher update (0-1)
        Higher gamma gives more weight to recent tasks
    device : str
        Device to store tensors
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        gamma: float = 1.0,
        device: str = 'cuda'
    ):
        super().__init__(model, ewc_lambda, device)
        self.gamma = gamma

    def consolidate(
        self,
        dataloader: DataLoader,
        num_samples: Optional[int] = None,
        **kwargs
    ):
        """
        Consolidate with exponential moving average

        F_new = γ * F_old + F_current
        """
        print(f"\nOnline consolidation (gamma={self.gamma})...")

        # Compute Fisher for current task
        current_fisher = self.compute_fisher(dataloader, num_samples)

        # Update Fisher with exponential moving average
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name in self.fisher_dict:
                    # Exponential moving average
                    self.fisher_dict[name] = (
                        self.gamma * self.fisher_dict[name] +
                        current_fisher[name]
                    )
                else:
                    # First task
                    self.fisher_dict[name] = current_fisher[name]

                # Update optimal parameters
                self.optimal_params[name] = param.data.clone().detach()

        self.consolidated_tasks += 1

        print(f"  Consolidated tasks: {self.consolidated_tasks}")


class SelectiveEWC(EWC):
    """
    Selective EWC

    Only protects parameters with high Fisher information,
    allowing less important parameters to adapt freely.

    This can improve plasticity while maintaining stability.

    Parameters:
    -----------
    model : nn.Module
        The neural network model
    ewc_lambda : float
        Importance weight
    selection_ratio : float
        Fraction of parameters to protect (0-1)
        e.g., 0.2 means protect top 20% most important parameters
    device : str
        Device
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        selection_ratio: float = 0.2,
        device: str = 'cuda'
    ):
        super().__init__(model, ewc_lambda, device)
        self.selection_ratio = selection_ratio
        self.selection_mask = {}

    def consolidate(self, dataloader: DataLoader, num_samples: Optional[int] = None, **kwargs):
        """Consolidate and compute selection mask"""
        super().consolidate(dataloader, num_samples, online=False)

        # Compute selection mask based on Fisher values
        all_fisher_values = []
        for name, fisher in self.fisher_dict.items():
            all_fisher_values.extend(fisher.flatten().tolist())

        # Find threshold for top selection_ratio
        threshold = torch.tensor(all_fisher_values).quantile(1 - self.selection_ratio)

        # Create masks
        for name, fisher in self.fisher_dict.items():
            self.selection_mask[name] = (fisher >= threshold).float()

        selected_params = sum(mask.sum().item() for mask in self.selection_mask.values())
        total_params = sum(f.numel() for f in self.fisher_dict.values())

        print(f"  Selected {selected_params}/{total_params} parameters ({self.selection_ratio*100:.1f}%)")

    def penalty(self) -> torch.Tensor:
        """Compute selective EWC penalty"""
        if self.consolidated_tasks == 0:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if name in self.fisher_dict:
                # Apply selection mask
                fisher = self.fisher_dict[name] * self.selection_mask[name]
                optimal = self.optimal_params[name]
                loss += (fisher * (param - optimal).pow(2)).sum()

        return (self.ewc_lambda / 2.0) * loss


if __name__ == "__main__":
    print("Testing EWC implementation...")

    # Create dummy model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 5)
    )

    # Test standard EWC
    ewc = EWC(model, ewc_lambda=1000.0, device='cpu')
    print(f"EWC lambda: {ewc.ewc_lambda}")
    print(f"Consolidated tasks: {ewc.consolidated_tasks}")

    # Test penalty before consolidation
    penalty = ewc.penalty()
    print(f"Penalty before consolidation: {penalty.item()}")

    # Test Online EWC
    online_ewc = OnlineEWC(model, ewc_lambda=1000.0, gamma=0.9, device='cpu')
    print(f"\nOnline EWC gamma: {online_ewc.gamma}")

    # Test Selective EWC
    selective_ewc = SelectiveEWC(model, ewc_lambda=1000.0, selection_ratio=0.3, device='cpu')
    print(f"Selective EWC ratio: {selective_ewc.selection_ratio}")

    print("\n✓ EWC implementations ready!")
