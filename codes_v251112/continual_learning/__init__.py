"""
Continual Learning Module for Multimodal Emotion Recognition

This module implements a complete continual learning framework for
multimodal (text, audio, video) emotion recognition with:

- AU-based emotion prediction via AU-EMO probability matrix
- Zero-shot learning for unseen emotion classes
- Multimodal consistency checking for pseudo-labeling
- Elastic Weight Consolidation (EWC) for anti-forgetting
- Domain splitting and task management
- Comprehensive evaluation metrics

Key Components:
---------------
- AUEMOMatrix: Bayesian-updatable AU-EMO probability associations
- AUEmotionNetwork: Neural network with AU prediction branch
- MultimodalConsistencyChecker: Validates unseen class predictions
- EWC: Prevents catastrophic forgetting
- ContinualLearningTrainer: Main training loop
- DomainSplitter: Creates task configurations
- ContinualLearningMetrics: Evaluation metrics

Example Usage:
--------------
```python
from continual_learning import (
    AUEmotionNetwork,
    ContinualLearningTrainer,
    DomainSplitter,
    create_predefined_task_sequence
)

# 1. Load AU-EMO prior
prior_matrix = load_au_emo_prior('au_emo_prior.json')

# 2. Create model
model = AUEmotionNetwork(
    text_input_dim=768,
    audio_input_dim=768,
    video_input_dim=768,
    num_aus=23,
    num_emotions=6,
    au_emo_prior=prior_matrix
)

# 3. Create trainer
trainer = ContinualLearningTrainer(
    model=model,
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-4),
    use_ewc=True
)

# 4. Create task sequence
tasks = create_predefined_task_sequence('custom', dataset_name='MOSEI')

# 5. Train on each task
splitter = DomainSplitter(dataset)
for task_config in tasks:
    seen_loader, unseen_loader = splitter.create_task_dataloaders(task_config)
    trainer.train_task(
        task_id=task_config.task_id,
        task_name=task_config.task_name,
        seen_loader=seen_loader,
        unseen_loader=unseen_loader,
        num_epochs=10
    )

# 6. Save and evaluate
trainer.save_final_model()
print(trainer.get_training_summary())
```

Version: 1.0.0
Author: openMM Team
"""

# Core components
from continual_learning.au_emo_matrix import (
    AUEMOMatrix,
    load_au_emo_prior
)

from continual_learning.au_emotion_network import (
    AUEmotionNetwork,
    AUPredictor,
    SingleModalityAUEmotionNetwork
)

from continual_learning.consistency_checker import (
    MultimodalConsistencyChecker,
    AdaptiveConsistencyChecker,
    ConsistencyStrategy
)

from continual_learning.ewc import (
    EWC,
    OnlineEWC,
    SelectiveEWC
)

from continual_learning.trainer import (
    ContinualLearningTrainer
)

from continual_learning.domain_splitter import (
    DomainSplitter,
    TaskConfig,
    create_predefined_task_sequence,
    EMOTION_NAMES,
    EMOTION_IDS
)

from continual_learning.metrics import (
    ContinualLearningMetrics
)

__all__ = [
    # AU-EMO Matrix
    'AUEMOMatrix',
    'load_au_emo_prior',

    # Network
    'AUEmotionNetwork',
    'AUPredictor',
    'SingleModalityAUEmotionNetwork',

    # Consistency
    'MultimodalConsistencyChecker',
    'AdaptiveConsistencyChecker',
    'ConsistencyStrategy',

    # Anti-forgetting
    'EWC',
    'OnlineEWC',
    'SelectiveEWC',

    # Training
    'ContinualLearningTrainer',

    # Domain management
    'DomainSplitter',
    'TaskConfig',
    'create_predefined_task_sequence',
    'EMOTION_NAMES',
    'EMOTION_IDS',

    # Metrics
    'ContinualLearningMetrics',
]

__version__ = '1.0.0'
