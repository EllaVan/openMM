# Blackbox Learnable Continual Learning Framework

## Overview

This directory contains the **blackbox** implementation of the continual learning framework using **end-to-end learnable parameters** for the AU-EMO matrix.

### Key Features

1. **End-to-End Learning**: Matrix optimized via gradient descent with rest of network
2. **Simplicity**: Single optimization pass, no EM algorithm
3. **Flexibility**: Can learn complex non-linear patterns
4. **Speed**: Faster training than whitebox Bayesian approach
5. **Extractable Probabilities**: Still provides interpretable probability matrix

## Mathematical Framework

### Learnable Matrix Model

The AU-EMO association is represented as a matrix of learnable logits:

```
Matrix M ∈ ℝ^(num_aus × num_emotions)
M_ij = learnable parameter (nn.Parameter)

Probability Conversion:
P(EMO_j|AU_i) = softmax_j(M_ij)
```

### Prediction

```
P(EMO_j|sample) = Σ_i P(EMO_j|AU_i) × P(AU_i|sample)
                = au_probs @ softmax(M, dim=1)
```

### Training

**Optimization**: End-to-end gradient descent

```
Loss = L_emotion + λ_reg × KL(P_current || P_prior) + λ_EWC × L_EWC

Where:
- L_emotion: Cross-entropy for emotion classification
- KL term: Regularization towards prior
- L_EWC: Elastic Weight Consolidation penalty
```

**Update Rule**:
```python
# Standard gradient descent
loss.backward()
optimizer.step()  # Updates both network and matrix
```

## Files

### 1. `learnable_matrix.py`

Learnable AU-EMO matrix implementation using nn.Parameter.

**Key Class**:
- `LearnableAUEMOMatrix`: Main matrix class
  - `matrix_logits`: nn.Parameter (learnable)
  - Initialized with psychology prior
  - Regularized to maintain connection to prior

**Key Methods**:
```python
# Get current probabilities
p_emo_given_au = matrix.get_probability_matrix()  # softmax(logits)

# Prediction (forward pass)
emo_pred = matrix(au_probs)  # au_probs @ P(EMO|AU)

# Regularization loss
reg_loss = matrix.compute_regularization_loss()  # KL to prior

# Statistics
stats = matrix.get_statistics()
```

### 2. `gradient_trainer.py`

Gradient descent trainer for continual learning.

**Key Class**:
- `GradientTrainerBlackbox`: Complete training orchestration

**Training Flow**:

**Task 0**:
1. Warmup: Train entire network (3 epochs)
2. Seen: Continue with regularization (10 epochs)
3. Unseen: Consistency checking + low-weight updates (5 epochs)
4. EWC: Consolidate Fisher information

**Task 1 to T**:
1. Seen: Train with EWC + matrix regularization
2. Unseen: Consistency checking + low-weight updates
3. EWC: Consolidate Fisher information

### 3. `blackbox_main.py`

Complete execution script from Task 0 to Task T.

**Usage**:
```bash
python blackbox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --task_sequence custom \
    --num_epochs 10 \
    --lr 1e-4 \
    --matrix_lr 1e-3 \
    --prior_strength 0.1 \
    --matrix_reg_lambda 0.1 \
    --seen_loss_weight 1.0 \
    --unseen_loss_weight 0.3 \
    --use_ewc \
    --ewc_lambda 1000.0 \
    --save_dir ../../checkpoints/blackbox
```

### 4. `README.md`

This file.

## Quick Start

### 1. Prepare AU-EMO Prior

Same as whitebox - create JSON file with P(AU|EMO) prior:

```json
{
  "au_names": ["AU1_Inner_Brow_Raiser", ...],
  "emotion_names": ["happy", "sad", "angry", "surprise", "disgust", "fear"],
  "prior_matrix": [
    [0.1, 0.2, 0.0, 0.3, 0.0, 0.1],
    ...
  ]
}
```

### 2. Run Training

```bash
cd codes_v251112/continual_learning/blackbox_learnable

python blackbox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path your_prior.json \
    --num_epochs 10 \
    --save_dir ../../checkpoints/blackbox
```

### 3. Monitor Results

Training outputs:
- `checkpoints/blackbox/task_*_checkpoint.pt`: Per-task checkpoints
- `checkpoints/blackbox/task_*_matrix.npz`: Per-task matrix states
- `checkpoints/blackbox/final_model.pt`: Final model
- `checkpoints/blackbox/final_matrix.npz`: Final matrix
- `checkpoints/blackbox/results/`: Performance plots
- `checkpoints/blackbox/training_summary.txt`: Complete summary

## Key Parameters

### Matrix Learning

```python
--lr 1e-4              # Learning rate for network
--matrix_lr 1e-3       # Learning rate for matrix
                       # Can be higher than network LR
                       # Matrix is smaller, can learn faster

--prior_strength 0.1   # Regularization strength (KL to prior)
                       # Higher = stay closer to prior
                       # Lower = more freedom to adapt

--matrix_reg_lambda 0.1  # Weight for regularization loss
                         # Balance between prior and data
```

### Loss Weighting

```python
--seen_loss_weight 1.0     # Full weight for seen (true labels)
--unseen_loss_weight 0.3   # Low weight for unseen (pseudo-labels)
                           # Much lower than whitebox because
                           # gradient descent is more sensitive
```

### EWC

```python
--use_ewc              # Enable EWC anti-forgetting
--ewc_lambda 1000.0    # EWC regularization strength
```

### Consistency Checking

```python
--consistency_strategy majority  # Require 3/4 modalities agree
--min_confidence 0.8            # Minimum confidence threshold
```

## Advantages of Blackbox Approach

### 1. Simplicity

- No EM algorithm needed
- Single optimization pass
- Easier to implement and debug

### 2. Speed

- Faster training than EM
- No alternating E-step/M-step
- Direct backpropagation

### 3. Flexibility

- Can learn arbitrary patterns
- Not constrained by probabilistic assumptions
- Better for complex relationships

### 4. Integration

- Seamlessly integrates with neural network
- Same optimizer for everything
- Unified training loop

### 5. Performance Potential

- May achieve better accuracy
- Can fit training data more closely
- More adaptive to new patterns

## Disadvantages

### 1. Interpretability

- Less transparent than Beta-Bernoulli
- Harder to explain individual parameters
- No built-in uncertainty quantification

### 2. Overfitting Risk

- Can overfit to training data
- Requires careful regularization
- May drift far from prior

### 3. Hyperparameter Sensitivity

- Performance depends on tuning
- Matrix LR, regularization strength, etc.
- Less theoretical guidance

### 4. Gradient Issues

- Potential gradient instability
- Can suffer from vanishing/exploding gradients
- Requires monitoring

## Output Interpretation

### Matrix Statistics

```
avg_probability: 0.1667        # Mean P(EMO|AU) (1/6 if uniform)
kl_from_prior: 0.2341          # KL divergence from prior
                               # Higher = more deviation
avg_entropy_per_au: 1.7854     # Avg entropy per AU
                               # Max = log(6) ≈ 1.79 (uniform)
                               # Low = overconfident
logits_mean: -0.1234           # Mean logit value
logits_std: 0.8765             # Logit standard deviation
```

### Understanding KL Divergence

Low KL (< 0.1):
- Matrix close to prior
- Strong regularization effect
- May underfit

Medium KL (0.1 - 0.5):
- Balanced adaptation
- Learning from data while respecting prior
- **Recommended range**

High KL (> 0.5):
- Significant deviation from prior
- May be overfitting
- Check if justified by performance

### Understanding Entropy

High entropy (close to log(num_emotions)):
- Uncertain predictions
- More uniform distribution
- May indicate insufficient training

Low entropy:
- Confident predictions
- Peaked distributions
- Check if overly confident (calibration)

### Logits Interpretation

```
logits_max - logits_min: range of logit values
Large range (> 5): Very confident differences
Small range (< 2): Uncertain, needs more training
```

## Comparison with Whitebox

| Aspect | Blackbox (Learnable) | Whitebox (Beta-Bernoulli) |
|--------|---------------------|---------------------------|
| Interpretability | ✗ Limited | ✓✓✓ Full transparency |
| Training speed | ✓✓✓ Fast (single pass) | ✗ Slower (EM) |
| Implementation | ✓✓✓ Simple | ✗ Complex (Bayesian) |
| Flexibility | ✓✓✓ High | ✗ Fixed update rule |
| Uncertainty | ✗ None | ✓✓✓ Exact variance |
| Overfitting risk | ✗ Higher | ✓ Lower (Bayesian) |
| Hyperparameter tuning | ✗ More sensitive | ✓ More robust |
| Prior integration | ✓ Via regularization | ✓✓✓ Principled Bayesian |

## Recommended Use Cases

**Use Blackbox When**:
- Performance is top priority
- Training speed matters
- Working with large datasets
- Complex non-linear patterns expected
- Production deployment (simpler codebase)
- Interpretability not critical

**Consider Whitebox When**:
- Need to explain model decisions
- Limited data (uncertainty helps)
- Working with domain experts
- Research/academic settings
- Regulatory requirements

## Hyperparameter Tuning Guide

### Matrix Learning Rate

Too high (> 1e-2):
- Matrix changes too rapidly
- Unstable training
- Poor generalization

**Recommended**: 1e-3 (10x higher than network LR)

Too low (< 1e-5):
- Matrix barely updates
- Stuck near prior
- Underutilizes learning

### Prior Strength

Too high (> 1.0):
- Matrix can't adapt
- Stuck at prior
- Poor performance on new domains

**Recommended**: 0.1 - 0.3

Too low (< 0.01):
- Matrix drifts far from prior
- Loses psychology knowledge
- May overfit

### Matrix Regularization Lambda

Too high (> 1.0):
- Regularization dominates
- Can't learn from data
- Matrix stays at prior

**Recommended**: 0.1 - 0.3

Too low (< 0.01):
- Insufficient regularization
- May overfit
- Unstable continual learning

### Unseen Loss Weight

Too high (> 0.5):
- Trusts pseudo-labels too much
- Error accumulation
- Performance degradation

**Recommended**: 0.2 - 0.4

Too low (< 0.1):
- Doesn't learn from unseen
- Poor zero-shot performance
- Wastes data

## Troubleshooting

### Issue: KL divergence exploding

**Symptoms**: KL from prior keeps increasing rapidly

**Solutions**:
1. Increase `--prior_strength` (e.g., 0.3)
2. Increase `--matrix_reg_lambda` (e.g., 0.3)
3. Reduce `--matrix_lr` (e.g., 5e-4)
4. Check if EWC is working

### Issue: Matrix not learning

**Symptoms**: KL from prior stays near 0, no performance improvement

**Solutions**:
1. Reduce `--prior_strength` (e.g., 0.05)
2. Reduce `--matrix_reg_lambda` (e.g., 0.05)
3. Increase `--matrix_lr` (e.g., 2e-3)
4. Check gradient flow

### Issue: Unseen performance poor

**Symptoms**: Low accuracy on unseen classes

**Solutions**:
1. Increase `--unseen_loss_weight` (e.g., 0.5)
2. Lower `--min_confidence` (e.g., 0.6)
3. Change `--consistency_strategy` to 'weighted_vote'
4. Ensure prior is good quality

### Issue: High forgetting

**Symptoms**: Previous task accuracy drops significantly

**Solutions**:
1. Increase `--ewc_lambda` (e.g., 5000)
2. Increase `--matrix_reg_lambda` (e.g., 0.5)
3. Reduce `--lr` for later tasks
4. Check EWC consolidation

### Issue: Training unstable

**Symptoms**: Loss oscillates wildly, NaN values

**Solutions**:
1. Reduce `--matrix_lr` (e.g., 1e-4)
2. Reduce `--lr` (e.g., 5e-5)
3. Add gradient clipping
4. Check for data issues

## Advanced: Extracting P(AU|EMO)

While the blackbox approach optimizes P(EMO|AU), you can estimate P(AU|EMO):

```python
# Load matrix
matrix = LearnableAUEMOMatrix(...)
matrix.load('final_matrix.npz')

# Get P(EMO|AU) (what model uses)
p_emo_given_au = matrix.get_probability_matrix()

# Estimate P(AU|EMO) (reverse Bayes)
p_au_given_emo = matrix.get_p_au_given_emo_estimate()

print("Estimated P(AU|EMO):")
print(p_au_given_emo)
```

Note: This is an approximation assuming uniform P(AU).

## Debugging Tips

### 1. Monitor Matrix Statistics

```python
stats = matrix.get_statistics()
print(f"KL from prior: {stats['kl_from_prior']:.4f}")
print(f"Entropy: {stats['avg_entropy_per_au']:.4f}")
```

Track these across training to ensure healthy evolution.

### 2. Visualize Matrix Evolution

```python
# After each task
print(matrix.visualize_matrix(
    au_names=au_names,
    emotion_names=emotion_names,
    show_logits=False
))
```

Look for gradual, sensible changes.

### 3. Check Gradient Norms

```python
# During training
grad_norm = matrix.matrix_logits.grad.norm().item()
print(f"Matrix gradient norm: {grad_norm:.6f}")
```

If too large (> 10): reduce matrix_lr
If too small (< 1e-6): increase matrix_lr

### 4. Compare Predictions

```python
# AU path vs direct path
print(f"AU path accuracy: {au_path_acc:.4f}")
print(f"Direct path accuracy: {direct_acc:.4f}")
```

If AU path much worse: matrix not learning properly
If similar: matrix is contributing

## Citation

If you use this blackbox framework, please cite:

```bibtex
@software{blackbox_continual_learning_2024,
  title={Blackbox Learnable Continual Learning for Emotion Recognition},
  author={openMM Team},
  year={2024},
  note={End-to-end learnable AU-EMO matrix}
}
```

## Contact

For questions or issues, please open a GitHub issue.
