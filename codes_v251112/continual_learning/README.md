# Multimodal Continual Learning for Emotion Recognition

A comprehensive framework for continual learning on multimodal (text, audio, video) emotion recognition with zero-shot capabilities.

## Overview

This module implements a novel continual learning approach that:

1. **Uses Action Units (AU) as intermediary representation** for emotion prediction
2. **Maintains a global AU-EMO probability matrix** updated across tasks
3. **Handles unseen emotion classes** through AU-based zero-shot prediction
4. **Prevents catastrophic forgetting** using Elastic Weight Consolidation (EWC)
5. **Validates pseudo-labels** through multimodal consistency checking

## Architecture

```
Multimodal Input (Text + Audio + Video)
    ↓
Multimodal Hypergraph Fusion
    ↓
    ├─→ AU Predictor (23 AUs)
    │       ↓
    │   AU-EMO Matrix (Global, Shared)
    │       ↓
    │   Emotion Prediction (via AU path)
    │
    └─→ Direct Emotion Classifier
            ↓
        Emotion Prediction (direct path)
```

### Key Components

1. **AU-EMO Matrix (`au_emo_matrix.py`)**
   - Bayesian-updatable probability matrix P(EMO|AU)
   - Initialized with psychology prior
   - Updated differently for seen (high weight) vs unseen (low weight) classes

2. **AU-Emotion Network (`au_emotion_network.py`)**
   - Extends MultimodalHypergraphFusion with AU prediction branch
   - Dual-path prediction: AU-based + direct
   - Integrated AU-EMO matrix for inference

3. **Consistency Checker (`consistency_checker.py`)**
   - Validates unseen class predictions across modalities
   - Strategies: All-agree, Majority, Weighted-vote, Entropy-threshold
   - Filters unreliable pseudo-labels

4. **EWC (`ewc.py`)**
   - Prevents forgetting of previous tasks
   - Variants: Standard, Online, Selective
   - No additional memory storage required

5. **Trainer (`trainer.py`)**
   - Manages multi-task training loop
   - Handles seen/unseen class separation
   - Integrates all components

6. **Domain Splitter (`domain_splitter.py`)**
   - Creates task configurations
   - Splits datasets into seen/unseen classes
   - Supports multiple splitting strategies

7. **Metrics (`metrics.py`)**
   - Average Accuracy
   - Forgetting Measure
   - Forward/Backward Transfer
   - Learning curves

## Installation

The module is part of the `openMM` project. Ensure dependencies are installed:

```bash
pip install torch torchvision numpy scikit-learn matplotlib tqdm
```

## Quick Start

### 1. Prepare AU-EMO Prior

Create a JSON file with your 23 AUs and their prior probabilities:

```json
{
  "au_names": ["AU1_Inner_Brow_Raiser", "AU2_Outer_Brow_Raiser", ...],
  "emotion_names": ["happy", "sad", "angry", "surprise", "disgust", "fear"],
  "prior_matrix": [
    [0.1, 0.2, 0.0, 0.3, 0.0, 0.1],  # AU1 probabilities
    [0.8, 0.1, 0.0, 0.6, 0.0, 0.2],  # AU2 probabilities
    ...
  ]
}
```

### 2. Load Dataset

```python
from hyper_fusion.dataloader import load_mosei_data

# Load your dataset
dataset = load_mosei_data(
    data_dir='./output/mosei_features',
    emotion='all'  # Load all emotions
)
```

### 3. Create Task Sequence

```python
from continual_learning import create_predefined_task_sequence

# Option A: Use predefined sequence
tasks = create_predefined_task_sequence('custom', dataset_name='MOSEI')

# Option B: Create custom tasks
from continual_learning import TaskConfig

tasks = [
    TaskConfig(0, "Task0", "MOSEI", seen_classes=[0, 1], unseen_classes=[2]),
    TaskConfig(1, "Task1", "MOSEI", seen_classes=[0], unseen_classes=[4, 5]),
    TaskConfig(2, "Task2", "MOSEI", seen_classes=[0, 1], unseen_classes=[3])
]
```

### 4. Initialize Model and Trainer

```python
import torch
from continual_learning import (
    AUEmotionNetwork,
    ContinualLearningTrainer,
    load_au_emo_prior
)

# Load AU-EMO prior
prior_matrix, au_names, emotion_names = load_au_emo_prior('au_emo_prior.json')

# Create model
model = AUEmotionNetwork(
    text_input_dim=768,
    audio_input_dim=768,
    video_input_dim=768,
    num_aus=23,
    num_emotions=6,
    au_emo_prior=prior_matrix,
    device='cuda'
)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Create trainer
trainer = ContinualLearningTrainer(
    model=model,
    optimizer=optimizer,
    device='cuda',
    use_ewc=True,
    ewc_lambda=1000.0,
    consistency_strategy='majority',
    save_dir='./checkpoints'
)
```

### 5. Train

```python
from continual_learning import DomainSplitter, ContinualLearningMetrics

# Create domain splitter
splitter = DomainSplitter(dataset, exclude_neutral=True)

# Create metrics tracker
metrics = ContinualLearningMetrics(num_tasks=len(tasks))

# Train on each task
for task_config in tasks:
    print(f"\n{'='*80}")
    print(f"Task {task_config.task_id}: {task_config.task_name}")
    print(f"{'='*80}")

    # Create dataloaders
    seen_loader, unseen_loader = splitter.create_task_dataloaders(
        task_config,
        batch_size=32,
        num_workers=4
    )

    # Train
    trainer.train_task(
        task_id=task_config.task_id,
        task_name=task_config.task_name,
        seen_loader=seen_loader,
        unseen_loader=unseen_loader,
        num_epochs=10
    )

    # Evaluate on all previous tasks
    for eval_task in range(task_config.task_id + 1):
        eval_seen_loader, _ = splitter.create_task_dataloaders(
            tasks[eval_task],
            batch_size=32
        )
        eval_results = trainer.evaluate(eval_seen_loader, phase=f'task_{eval_task}')

        if 'accuracy' in eval_results:
            metrics.update(
                task_trained=task_config.task_id,
                task_eval=eval_task,
                predictions=eval_results['predictions'],
                labels=eval_results['labels']
            )

# Save final model
trainer.save_final_model()

# Print results
print(trainer.get_training_summary())
print(metrics.get_summary())

# Plot results
metrics.plot_performance_matrix(save_path='./results/performance_matrix.png')
metrics.plot_learning_curves(save_path='./results/learning_curves.png')
```

## Configuration Options

### AU-EMO Matrix

```python
from continual_learning import AUEMOMatrix

matrix = AUEMOMatrix(
    num_aus=23,
    num_emotions=6,
    prior_matrix=prior,
    prior_strength=100.0,  # Higher = more resistant to updates
    device='cuda'
)

# Update with seen class
matrix.update(
    au_probs=au_predictions,
    emo_labels=true_labels,
    is_seen=True,
    seen_weight=10.0  # High weight for seen classes
)

# Update with unseen class (filtered by confidence)
matrix.update(
    au_probs=au_predictions,
    emo_labels=pseudo_labels,
    is_seen=False,
    confidence=confidence_scores,
    unseen_weight=1.0,  # Low weight for unseen
    min_confidence=0.8  # Only update if confidence > 0.8
)
```

### Consistency Checking

```python
from continual_learning import MultimodalConsistencyChecker, ConsistencyStrategy

checker = MultimodalConsistencyChecker(
    model=model,
    strategy=ConsistencyStrategy.MAJORITY,  # or ALL_AGREE, WEIGHTED_VOTE, etc.
    min_confidence=0.8,
    entropy_threshold=0.5
)

result = checker.check_consistency(text, audio, video, masks)
# result['is_consistent']: [batch_size] bool
# result['consensus_label']: [batch_size] int
# result['confidence']: [batch_size] float
```

### EWC Configuration

```python
from continual_learning import OnlineEWC

ewc = OnlineEWC(
    model=model,
    ewc_lambda=1000.0,  # Regularization strength
    gamma=0.9,          # Decay factor for online Fisher
    device='cuda'
)

# After each task
ewc.consolidate(seen_loader, num_samples=1000)

# During training
loss = classification_loss + ewc.penalty()
```

## Advanced Usage

### Custom Task Splitting Strategy

```python
splitter = DomainSplitter(dataset)

# Create tasks with custom strategy
tasks = splitter.create_tasks_by_strategy(
    strategy='small_unseen',  # or 'incremental', 'disjoint', 'overlap'
    num_tasks=5,
    seen_classes_base=[0, 1]  # happy, sad
)

# Save/load task configurations
splitter.save_task_configs(tasks, 'task_configs.json')
tasks = DomainSplitter.load_task_configs('task_configs.json')
```

### Adaptive Consistency Threshold

```python
from continual_learning import AdaptiveConsistencyChecker

checker = AdaptiveConsistencyChecker(
    model=model,
    initial_confidence=0.7,
    adaptation_rate=0.1,
    target_consistency_rate=0.3  # Aim for 30% consistent samples
)
# Automatically adjusts threshold based on observed consistency rate
```

### Visualizing AU-EMO Matrix

```python
# Get matrix visualization
viz = model.au_emo_matrix.visualize_matrix(
    au_names=au_names,
    emotion_names=emotion_names
)
print(viz)

# Get statistics
stats = model.get_au_emo_statistics()
print(f"Total updates: {stats['total_updates']}")
print(f"Matrix entropy: {stats['matrix_entropy']:.4f}")
```

## Expected Results

Based on the framework design:

- **Seen classes**: 80-95% accuracy (depends on dataset)
- **Unseen classes**: 60-75% accuracy (zero-shot via AU-EMO matrix)
- **Forgetting**: < 10% drop on previous tasks (with EWC)
- **Consistency rate**: 20-40% of unseen samples pass consistency check

## File Structure

```
continual_learning/
├── __init__.py                  # Module exports
├── README.md                    # This file
├── au_emo_matrix.py            # AU-EMO probability matrix
├── au_emotion_network.py       # Neural network with AU branch
├── consistency_checker.py      # Multimodal consistency validation
├── ewc.py                      # Elastic Weight Consolidation
├── trainer.py                  # Main training loop
├── domain_splitter.py          # Task configuration and data splitting
└── metrics.py                  # Evaluation metrics
```

## Citation

If you use this code, please cite:

```bibtex
@software{openMM_continual_learning,
  title={Multimodal Continual Learning for Emotion Recognition},
  author={openMM Team},
  year={2024},
  url={https://github.com/EllaVan/openMM}
}
```

## Troubleshooting

### Issue: AU-EMO matrix not updating

**Solution**: Check that `au_probs` values are in [0, 1] range (use sigmoid activation)

### Issue: Consistency rate too low

**Solution**:
- Lower `min_confidence` threshold
- Use `ConsistencyStrategy.MAJORITY` instead of `ALL_AGREE`
- Use `AdaptiveConsistencyChecker`

### Issue: High forgetting on previous tasks

**Solution**:
- Increase `ewc_lambda` (e.g., from 1000 to 5000)
- Use `OnlineEWC` instead of standard EWC
- Increase `au_emo_regularization` strength

### Issue: Poor unseen class performance

**Solution**:
- Verify AU-EMO prior quality
- Check consistency checker settings
- Increase `unseen_update_weight` (but not too high to avoid noise)

## License

MIT License

## Contact

For questions and issues, please open an issue on GitHub.
