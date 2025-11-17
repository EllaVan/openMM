"""
Knowledge Distillation for Continual Learning

Implements various knowledge distillation strategies to preserve knowledge
from previous tasks while learning new ones.

Key Features:
1. Response-based distillation (output logits)
2. Feature-based distillation (intermediate representations)
3. AU-based distillation (AU activation patterns)
4. Multi-teacher distillation (from multiple previous tasks)

References:
- Learning without Forgetting (LwF): Li & Hoiem, 2016
- Knowledge Distillation: Hinton et al., 2015
- DILLB approach for domain incremental learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import copy


class KnowledgeDistillationLoss(nn.Module):
    """
    Knowledge Distillation Loss

    Computes KL divergence between student and teacher predictions
    with temperature scaling.

    Loss = KL(softmax(teacher_logits / T) || softmax(student_logits / T)) * T^2

    The T^2 factor ensures gradients remain appropriately scaled.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        """
        Args:
            temperature: Softening parameter (higher = softer distributions)
            alpha: Weight for distillation loss (1-alpha for task loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distillation loss

        Args:
            student_logits: [batch, num_classes]
            teacher_logits: [batch, num_classes]

        Returns:
            distillation_loss: scalar
        """
        T = self.temperature

        # Soften probabilities
        student_soft = F.log_softmax(student_logits / T, dim=1)
        teacher_soft = F.softmax(teacher_logits / T, dim=1)

        # KL divergence with temperature scaling
        kl_div = F.kl_div(
            student_soft,
            teacher_soft,
            reduction='batchmean'
        ) * (T * T)

        return kl_div


class FeatureDistillationLoss(nn.Module):
    """
    Feature-level Distillation Loss

    Minimizes difference between intermediate feature representations
    of student and teacher networks.

    Common choices:
    - L2 distance: ||student_features - teacher_features||^2
    - Cosine similarity: 1 - cos(student, teacher)
    """

    def __init__(self, distance_type: str = 'l2'):
        """
        Args:
            distance_type: 'l2', 'cosine', or 'huber'
        """
        super().__init__()
        self.distance_type = distance_type

    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature distillation loss

        Args:
            student_features: [batch, feature_dim]
            teacher_features: [batch, feature_dim]

        Returns:
            feature_loss: scalar
        """
        if self.distance_type == 'l2':
            loss = F.mse_loss(student_features, teacher_features)

        elif self.distance_type == 'cosine':
            # Cosine similarity loss
            cos_sim = F.cosine_similarity(
                student_features, teacher_features, dim=1
            ).mean()
            loss = 1 - cos_sim

        elif self.distance_type == 'huber':
            loss = F.smooth_l1_loss(student_features, teacher_features)

        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

        return loss


class AUDistillationLoss(nn.Module):
    """
    AU-level Distillation Loss

    Preserves AU activation patterns from teacher to student.
    Since AUs are more stable across domains than emotions,
    this helps maintain consistent intermediate representations.
    """

    def __init__(self, loss_type: str = 'bce'):
        """
        Args:
            loss_type: 'bce' (binary cross-entropy) or 'mse'
        """
        super().__init__()
        self.loss_type = loss_type

    def forward(
        self,
        student_au_probs: torch.Tensor,
        teacher_au_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute AU distillation loss

        Args:
            student_au_probs: [batch, num_aus]
            teacher_au_probs: [batch, num_aus]

        Returns:
            au_loss: scalar
        """
        if self.loss_type == 'bce':
            # Binary cross-entropy (treating teacher as soft targets)
            loss = F.binary_cross_entropy(
                student_au_probs,
                teacher_au_probs.detach(),
                reduction='mean'
            )

        elif self.loss_type == 'mse':
            # Mean squared error
            loss = F.mse_loss(student_au_probs, teacher_au_probs.detach())

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss


class MultiTaskDistillationManager:
    """
    Multi-Task Distillation Manager

    Manages knowledge distillation from multiple previous tasks.

    Strategy:
    1. Store teacher models for each completed task
    2. During new task training, distill from all relevant teachers
    3. Weight distillation losses based on task similarity or recency
    """

    def __init__(
        self,
        temperature: float = 2.0,
        alpha_kd: float = 0.3,
        alpha_feature: float = 0.2,
        alpha_au: float = 0.1,
        device: str = 'cuda'
    ):
        """
        Args:
            temperature: Distillation temperature
            alpha_kd: Weight for response distillation
            alpha_feature: Weight for feature distillation
            alpha_au: Weight for AU distillation
            device: Device for computations
        """
        self.temperature = temperature
        self.alpha_kd = alpha_kd
        self.alpha_feature = alpha_feature
        self.alpha_au = alpha_au
        self.device = device

        # Loss modules
        self.kd_loss = KnowledgeDistillationLoss(temperature=temperature)
        self.feature_loss = FeatureDistillationLoss(distance_type='l2')
        self.au_loss = AUDistillationLoss(loss_type='bce')

        # Teacher models (one per task)
        self.teachers = {}

        # Task-specific weights (for weighted distillation)
        self.task_weights = {}

    def add_teacher(self, task_id: str, teacher_model: nn.Module):
        """
        Add a teacher model for a completed task

        Args:
            task_id: Task identifier
            teacher_model: Trained model to use as teacher
        """
        # Deep copy to avoid reference issues
        teacher = copy.deepcopy(teacher_model)
        teacher.eval()

        # Freeze all parameters
        for param in teacher.parameters():
            param.requires_grad = False

        self.teachers[task_id] = teacher.to(self.device)
        self.task_weights[task_id] = 1.0  # Default equal weight

        print(f"Added teacher for task: {task_id}")

    def set_task_weight(self, task_id: str, weight: float):
        """Set importance weight for a specific teacher"""
        if task_id in self.task_weights:
            self.task_weights[task_id] = weight
            print(f"Set weight for task {task_id}: {weight}")

    def compute_distillation_loss(
        self,
        student_model: nn.Module,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        masks: Optional[Dict] = None,
        current_task_id: str = None,
        distill_from_tasks: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task distillation loss

        Args:
            student_model: Current model being trained
            text, audio, video: Input data
            masks: Optional masks
            current_task_id: Current task being trained
            distill_from_tasks: Which tasks to distill from (None = all)

        Returns:
            dict with loss components:
                - total_distill_loss
                - kd_loss
                - feature_loss
                - au_loss
                - per_task_losses
        """
        if len(self.teachers) == 0:
            # No teachers yet (Task 0)
            return {
                'total_distill_loss': torch.tensor(0.0, device=self.device),
                'kd_loss': torch.tensor(0.0, device=self.device),
                'feature_loss': torch.tensor(0.0, device=self.device),
                'au_loss': torch.tensor(0.0, device=self.device)
            }

        # Determine which teachers to use
        if distill_from_tasks is None:
            distill_from_tasks = list(self.teachers.keys())

        # Student forward pass
        student_model.eval()  # Eval mode for distillation
        with torch.no_grad():
            student_outputs = student_model(
                text, audio, video, masks,
                domain_id=current_task_id,
                return_features=True
            )

        student_model.train()  # Back to train mode

        # Accumulate losses from all teachers
        total_kd_loss = 0.0
        total_feature_loss = 0.0
        total_au_loss = 0.0
        per_task_losses = {}

        total_weight = sum(self.task_weights[tid] for tid in distill_from_tasks)

        for teacher_task_id in distill_from_tasks:
            teacher = self.teachers[teacher_task_id]
            weight = self.task_weights[teacher_task_id] / total_weight

            # Teacher forward pass
            with torch.no_grad():
                teacher_outputs = teacher(
                    text, audio, video, masks,
                    domain_id=teacher_task_id,
                    return_features=True
                )

            # 1. Response distillation (emotion logits)
            if student_outputs['emo_from_au'] is not None:
                kd = self.kd_loss(
                    student_outputs['emo_from_au'],
                    teacher_outputs['emo_from_au']
                )
                total_kd_loss += weight * kd

            # 2. Feature distillation (multimodal features)
            if 'multimodal_features' in student_outputs:
                feat = self.feature_loss(
                    student_outputs['multimodal_features'],
                    teacher_outputs['multimodal_features']
                )
                total_feature_loss += weight * feat

            # 3. AU distillation
            au = self.au_loss(
                student_outputs['au_probs'],
                teacher_outputs['au_probs']
            )
            total_au_loss += weight * au

            # Track per-task loss
            per_task_losses[teacher_task_id] = {
                'kd': kd.item() if isinstance(kd, torch.Tensor) else 0.0,
                'feature': feat.item() if isinstance(feat, torch.Tensor) else 0.0,
                'au': au.item()
            }

        # Weighted combination
        total_distill_loss = (
            self.alpha_kd * total_kd_loss +
            self.alpha_feature * total_feature_loss +
            self.alpha_au * total_au_loss
        )

        return {
            'total_distill_loss': total_distill_loss,
            'kd_loss': total_kd_loss,
            'feature_loss': total_feature_loss,
            'au_loss': total_au_loss,
            'per_task_losses': per_task_losses
        }

    def clear_teachers(self):
        """Clear all teacher models (free memory)"""
        self.teachers.clear()
        self.task_weights.clear()
        print("All teachers cleared")


def test_knowledge_distillation():
    """Test knowledge distillation modules"""
    print("Testing Knowledge Distillation...")

    # Test basic KD loss
    kd_loss = KnowledgeDistillationLoss(temperature=2.0)

    student_logits = torch.randn(4, 6)
    teacher_logits = torch.randn(4, 6)

    loss = kd_loss(student_logits, teacher_logits)
    print(f"\nKD Loss: {loss.item():.4f}")

    # Test feature distillation
    feat_loss = FeatureDistillationLoss(distance_type='l2')

    student_feat = torch.randn(4, 768)
    teacher_feat = torch.randn(4, 768)

    loss = feat_loss(student_feat, teacher_feat)
    print(f"Feature Loss (L2): {loss.item():.4f}")

    # Test AU distillation
    au_loss = AUDistillationLoss(loss_type='bce')

    student_au = torch.sigmoid(torch.randn(4, 23))
    teacher_au = torch.sigmoid(torch.randn(4, 23))

    loss = au_loss(student_au, teacher_au)
    print(f"AU Loss (BCE): {loss.item():.4f}")

    print("\n✓ Knowledge distillation test passed!")


if __name__ == "__main__":
    test_knowledge_distillation()