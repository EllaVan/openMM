# Whitebox Bayesian Continual Learning Framework

## Overview

This directory contains the **whitebox** implementation of the continual learning framework using **Beta-Bernoulli conjugate prior** for interpretable P(AU|EMO) probability updating.

### Key Features

1. **Full Interpretability**: Each P(AU|EMO) is modeled as Beta(α, β) with transparent statistical meaning
2. **Uncertainty Quantification**: Beta distribution provides variance estimates for each probability
3. **Principled Bayesian Updates**: Conjugate prior guarantees mathematically sound updates
4. **EM Algorithm**: Avoids circular dependency between AU predictor and matrix updates

## Mathematical Framework

### Beta-Bernoulli Model

For each AU-EMO pair (i, j):

```
P(AU_i|EMO_j) ~ Beta(α_ij, β_ij)

Point Estimate:
  P(AU_i=1|EMO_j) = α_ij / (α_ij + β_ij)

Uncertainty:
  Var[P] = αβ / [(α+β)²(α+β+1)]
```

### Bayesian Update Rule

When observing sample with emotion EMO_j and AU activation a_i:

```
If AU_i is active (a_i ≈ 1):
  α_ij += weight × a_i

If AU_i is inactive (a_i ≈ 0):
  β_ij += weight × (1 - a_i)
```

### EM Algorithm

**E-step** (Expectation):
- Freeze Beta parameters (α, β)
- Optimize AU predictor network
- Loss: emotion classification via AU-EMO matrix

**M-step** (Maximization):
- Freeze AU predictor
- Collect AU predictions from network
- Update Beta parameters using Bayesian rule

## Files

### 1. `beta_bernoulli_matrix.py`

Beta-Bernoulli AU-EMO probability matrix implementation.

**Key Classes**:
- `BetaBernoulliAUEMOMatrix`: Main matrix class
  - Stores α, β parameters for each AU-EMO pair
  - Provides P(AU|EMO) and P(EMO|AU) estimates
  - Implements Bayesian updates
  - Tracks uncertainty

**Key Methods**:
```python
# Get current probabilities
p_au_given_emo = matrix.get_p_au_given_emo()  # For interpretation
p_emo_given_au = matrix.get_p_emo_given_au()  # For prediction

# Get uncertainty
uncertainty = matrix.get_uncertainty()  # Variance

# Update from observations
matrix.update_from_labels(
    au_probs=au_predictions,
    emo_labels=labels,
    is_seen=True,
    seen_weight=1.0
)

# Regularize towards prior
matrix.regularize_to_prior(strength=0.01)
```

### 2. `em_trainer.py`

EM algorithm trainer for continual learning.

**Key Class**:
- `EMTrainerWhitebox`: Complete training orchestration

**Training Flow**:

**Task 0**:
1. Warmup: Train AU predictor (3 epochs)
2. Seen EM: Alternate E-step and M-step (10 epochs × 3 EM iterations)
3. Unseen: Consistency checking + matrix update (5 epochs)
4. EWC: Consolidate Fisher information

**Task 1 to T**:
1. Seen EM: E-step and M-step with EWC penalty
2. Unseen: Consistency checking + matrix update with validation
3. EWC: Consolidate Fisher information

### 3. `whitebox_main.py`

Complete execution script from Task 0 to Task T.

**Usage**:
```bash
python whitebox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path ../example_au_emo_prior.json \
    --task_sequence custom \
    --num_epochs 10 \
    --num_em_iterations 3 \
    --prior_strength 100.0 \
    --seen_update_weight 1.0 \
    --unseen_update_weight 0.8 \
    --use_ewc \
    --ewc_lambda 1000.0 \
    --save_dir ../../checkpoints/whitebox
```

### 4. `README.md`

This file.

## Quick Start

### 1. Prepare AU-EMO Prior

Create a JSON file with P(AU|EMO) prior:

```json
{
  "au_names": ["AU1_Inner_Brow_Raiser", "AU2_Outer_Brow_Raiser", ...],
  "emotion_names": ["happy", "sad", "angry", "surprise", "disgust", "fear"],
  "prior_matrix": [
    [0.1, 0.2, 0.0, 0.3, 0.0, 0.1],
    [0.2, 0.4, 0.1, 0.5, 0.0, 0.2],
    ...
  ]
}
```

### 2. Run Training

```bash
cd codes_v251112/continual_learning/whitebox_bayesian

python whitebox_main.py \
    --data_dir ../../output/mosei_features \
    --au_prior_path your_prior.json \
    --num_epochs 10 \
    --save_dir ../../checkpoints/whitebox
```

### 3. Monitor Results

Training outputs:
- `checkpoints/whitebox/task_*_checkpoint.pt`: Per-task checkpoints
- `checkpoints/whitebox/task_*_matrix.npz`: Per-task matrix states
- `checkpoints/whitebox/final_model.pt`: Final model
- `checkpoints/whitebox/final_matrix.npz`: Final matrix
- `checkpoints/whitebox/results/`: Performance plots
- `checkpoints/whitebox/training_summary.txt`: Complete summary

## Key Parameters

### Beta-Bernoulli Matrix

```python
--prior_strength 100.0      # Total pseudo-count per AU-EMO pair
                            # Higher = harder to update from prior
                            # Lower = more adaptive to data

--seen_update_weight 1.0    # Weight for seen class updates
                            # Full trust in true labels

--unseen_update_weight 0.8  # Weight for unseen class updates
                            # Lower because pseudo-labels less reliable

--au_emo_regularization 0.01  # Regularization strength towards prior
                               # Applied every 5 epochs
```

### EM Algorithm

```python
--num_em_iterations 3  # Number of E-step/M-step cycles per epoch
                       # More iterations = better convergence
                       # But slower training
```

### EWC

```python
--use_ewc              # Enable EWC anti-forgetting
--ewc_lambda 1000.0    # EWC regularization strength
                       # Higher = less forgetting, harder to learn new tasks
```

### Consistency Checking

```python
--consistency_strategy majority  # Require 3/4 modalities to agree
--min_confidence 0.8            # Minimum confidence for unseen updates
```

## Advantages of Whitebox Approach

### 1. Full Interpretability

- Every parameter has clear statistical meaning
- α_ij: Pseudo-count of AU_i being active for EMO_j
- β_ij: Pseudo-count of AU_i being inactive for EMO_j
- Can explain every probability estimate

### 2. Uncertainty Quantification

```python
uncertainty = matrix.get_uncertainty()
# High uncertainty → Need more data
# Low uncertainty → Confident estimate
```

### 3. Principled Updates

- Conjugate prior guarantees well-behaved updates
- No gradient instability
- Convergence guarantees from Bayesian theory

### 4. Prior Knowledge Integration

- Psychology prior seamlessly integrated
- Prior strength controls influence
- Can never completely lose prior knowledge

### 5. Inspection and Debugging

```python
# Visualize matrix with uncertainties
print(matrix.visualize_matrix(show_uncertainty=True))

# Track statistics
stats = matrix.get_statistics()
# - effective_sample_size: How much data matrix has seen
# - kl_from_prior: How much matrix has deviated from prior
# - avg_uncertainty: Overall confidence level
```

## Disadvantages

### 1. Assumptions

- Assumes AU activations are Bernoulli (binary)
- Assumes uniform P(EMO) for conversion to P(EMO|AU)
- May not capture complex dependencies

### 2. Flexibility

- Fixed update rule (Bayesian)
- Cannot learn non-linear transformations
- Limited by model structure

### 3. Computational Cost

- EM algorithm requires multiple passes
- E-step and M-step separation increases training time

## Output Interpretation

### Matrix Statistics

```
total_updates: 12500          # Total samples used for updates
seen_updates: 10000           # Seen class samples
unseen_updates: 2500          # Unseen class samples
avg_p_au_given_emo: 0.3421    # Average activation probability
avg_uncertainty: 0.0023       # Average variance (lower = more confident)
effective_sample_size: 142.5  # Average α+β (higher = more data)
kl_from_prior: 0.1234         # KL divergence from prior (lower = closer to prior)
```

### Understanding Uncertainty

Low uncertainty (< 0.001):
- Confident estimate
- Sufficient data observed
- Trustworthy probability

High uncertainty (> 0.01):
- Uncertain estimate
- Limited data
- May need more samples

### Regularization Effects

Before regularization:
```
kl_from_prior: 0.5234  # Drifted far from prior
```

After regularization (strength=0.01):
```
kl_from_prior: 0.4827  # Pulled slightly back towards prior
```

## Comparison with Blackbox

| Aspect | Whitebox (Beta-Bernoulli) | Blackbox (Learnable) |
|--------|---------------------------|----------------------|
| Interpretability | ✓✓✓ Full transparency | ✗ Limited |
| Uncertainty | ✓✓✓ Exact variance | ✗ No built-in estimates |
| Prior integration | ✓✓✓ Principled Bayesian | ✓ Via regularization |
| Flexibility | ✗ Fixed update rule | ✓✓✓ End-to-end learning |
| Training speed | ✗ Slower (EM) | ✓✓✓ Faster (single pass) |
| Stability | ✓✓✓ Guaranteed convergence | ✓ Depends on hyperparams |
| Debugging | ✓✓✓ Easy to inspect | ✗ Black box |

## Recommended Use Cases

**Use Whitebox When**:
- Interpretability is critical
- Need to explain model decisions
- Working with domain experts (psychologists)
- Limited data (uncertainty quantification helps)
- Research/academic settings
- Regulatory requirements for transparency

**Consider Blackbox When**:
- Performance is top priority
- Training speed matters
- Complex non-linear relationships expected
- Large datasets available
- Production deployment without explanation needs

## Troubleshooting

### Issue: Matrix not updating

**Check**:
1. `effective_sample_size` increasing? → Matrix is updating
2. `kl_from_prior` changing? → Matrix evolving
3. Confidence too low? → Increase `--unseen_update_weight`

### Issue: High forgetting

**Solutions**:
1. Increase `--ewc_lambda` (e.g., 5000)
2. Reduce `--au_emo_regularization` (less drift from prior)
3. Increase `--prior_strength` (stronger prior anchoring)

### Issue: Poor unseen performance

**Solutions**:
1. Lower `--min_confidence` (e.g., 0.6-0.7)
2. Change `--consistency_strategy` to 'weighted_vote'
3. Increase `--num_em_iterations`

### Issue: Slow training

**Solutions**:
1. Reduce `--num_em_iterations` (e.g., 2)
2. Reduce `--num_epochs` per task
3. Increase `--batch_size`

## Citation

If you use this whitebox framework, please cite:

```bibtex
@software{whitebox_continual_learning_2024,
  title={Whitebox Bayesian Continual Learning for Emotion Recognition},
  author={openMM Team},
  year={2024},
  note={Beta-Bernoulli AU-EMO matrix with EM algorithm}
}
```

## Contact

For questions or issues, please open a GitHub issue.
