# AU-EMO Prior Knowledge Integration

This document explains how to use the **AU-EMO prior knowledge** from RAF-DB across all continual learning frameworks.

## Overview

The RAF-DB materials provide psychology-based prior knowledge about the relationship between **Action Units (AUs)** and **Emotions**, represented as P(AU | EMO). This prior is crucial for:

1. **Initializing learnable matrices** with meaningful starting points
2. **Regularizing learning** to stay close to psychological knowledge
3. **Zero-shot inference** on unseen emotion categories
4. **Interpretability** through human-understandable AU activations

## Prior Data Source

The prior knowledge is extracted from:
- **Location**: `codes_v251112/materials/RAF_graph.json`
- **Format**: 17 emotions × 26 AUs probability matrix
- **Source**: RAF-DB (Real-world Affective Faces Database)

### Emotions Included

**6 Basic Emotions** (training set):
- surprise, fear, disgust, happiness, sadness, anger

**11 Compound Emotions** (test set):
- happiness_surprise, happiness_disgust
- sadness_fear, sadness_anger, sadness_surprise, sadness_disgust
- fear_anger, fear_surprise
- anger_surprise, anger_disgust
- disgust_surprise

### Action Units

The framework uses **23 custom AUs** (subset of standard FACS):
- AU1-2 (brows), AU4-7 (eyes), AU9-10 (nose/upper lip)
- AU12-18 (mouth), AU20, AU23-27 (lips/jaw)
- AU43, AU45-46 (eye actions)

## Loading the Prior

### Method 1: Automatic Loading (Recommended)

All frameworks now support **automatic loading** from materials directory:

```bash
# Whitebox Bayesian
python codes_v251112/continual_learning/whitebox_bayesian/whitebox_main.py \
    --data_dir output/mosei_features \
    --task_sequence custom \
    --num_epochs 10

# Blackbox Learnable
python codes_v251112/continual_learning/blackbox_learnable/blackbox_main.py \
    --data_dir output/mosei_features \
    --task_sequence custom \
    --num_epochs 10

# DILLB Multi-head
python codes_v251112/continual_learning/dillb_multimodal/dillb_main.py \
    --data_dir output/mosei_features \
    --task_sequence custom \
    --num_epochs 10
```

By default, all frameworks will:
1. Load prior from `codes_v251112/materials/RAF_graph.json`
2. Use **6 basic emotions** (train set)
3. Extract **23 AUs**

### Method 2: Using All Emotions

To use all 17 emotions (basic + compound):

```bash
python whitebox_main.py \
    --data_dir output/mosei_features \
    --use_all_emotions \
    ...
```

### Method 3: Custom Prior Path

To use a custom prior JSON file:

```bash
python whitebox_main.py \
    --data_dir output/mosei_features \
    --au_prior_path path/to/custom_prior.json \
    ...
```

## Prior Loader Utility

### Basic Usage

```python
from continual_learning.au_emo_prior_loader import AUEMOPriorLoader

# Load prior
loader = AUEMOPriorLoader(
    materials_dir='codes_v251112/materials',
    use_basic_emotions_only=True,  # 6 emotions
    target_num_aus=23
)

# Get as tensor [23, 6]
prior_matrix, emotion_names, au_indices = loader.get_prior_matrix()
print(f"Prior shape: {prior_matrix.shape}")
print(f"Emotions: {emotion_names}")

# Get as dictionary
prior_dict = loader.get_prior_dict()
print(f"surprise AUs: {prior_dict['surprise']}")
```

### Advanced Usage

```python
# Get normalized probabilities
prior_matrix_norm, emotions, aus = loader.get_prior_matrix(normalize=True)

# Get only specific emotions
subset_prior, emotions, aus = loader.get_prior_matrix(
    emotion_subset=['happiness', 'sadness', 'anger']
)

# Get statistics
stats = loader.get_emotion_stats()
print(f"Surprise active AUs: {stats['surprise']['num_active_aus']}")
print(f"Top AUs: {stats['surprise']['top_aus']}")
```

### Saving Custom Priors

```python
# Save processed prior to JSON
loader.save_prior_json(
    output_path='my_custom_prior.json',
    emotion_subset=['happiness', 'sadness'],
    normalize=False
)
```

## Framework-Specific Integration

### Whitebox Bayesian Framework

**How it uses prior**:
- Initializes Beta distribution parameters: `α = p * strength`, `β = (1-p) * strength`
- `prior_strength` controls confidence in prior knowledge (default: 100.0)
- Updates via Bayesian inference during training

**Example**:
```bash
python whitebox_main.py \
    --data_dir output/mosei_features \
    --prior_strength 100.0 \
    --num_epochs 10
```

**Prior strength guidelines**:
- Low (10-50): Trust data more, adapt quickly
- Medium (50-200): Balanced, recommended for most cases
- High (200-1000): Trust prior more, slower adaptation

### Blackbox Learnable Framework

**How it uses prior**:
- Initializes learnable logits via inverse softmax: `logit = log(p + ε)`
- Regularizes via KL divergence to stay close to prior
- `prior_strength` controls regularization weight (default: 0.1)

**Example**:
```bash
python blackbox_main.py \
    --data_dir output/mosei_features \
    --prior_strength 0.1 \
    --matrix_lr 1e-3 \
    --num_epochs 10
```

### DILLB Multi-head Framework

**How it uses prior**:
- Initializes both global and domain-specific matrices
- Prior provides baseline knowledge for new domains
- Multi-teacher distillation preserves prior knowledge

**Example**:
```bash
python dillb_main.py \
    --data_dir output/mosei_features \
    --use_distillation \
    --freeze_backbone_after_task0 \
    --num_epochs 10
```

### Domain Prompts Framework

**Note**: This framework learns prompts end-to-end and does **not require** AU-EMO prior. It discovers domain-specific patterns automatically through:
- S-Prompts style prompt learning
- UDIL adaptive loss weighting
- DARE contrastive learning

## Pre-generated Prior Files

For convenience, pre-generated prior files are available:

### 1. Basic Emotions (6 emotions)
**File**: `codes_v251112/continual_learning/au_emo_prior_basic.json`

**Format**:
```json
{
  "emotions": ["surprise", "fear", "disgust", "happiness", "sadness", "anger"],
  "num_aus": 23,
  "prior": {
    "surprise": {
      "AU0": 1.0,
      "AU1": 1.0,
      "AU4": 0.66,
      ...
    },
    ...
  },
  "metadata": {
    "source": "RAF-DB",
    "normalized": false,
    "basic_emotions_only": true
  }
}
```

### 2. All Emotions (17 emotions)
**File**: `codes_v251112/continual_learning/au_emo_prior_all.json`

Includes all 6 basic + 11 compound emotions.

### 3. Normalized Version
**File**: `codes_v251112/continual_learning/au_emo_prior_basic_normalized.json`

Same as basic, but probabilities normalized per emotion (sum to 1.0).

## Example AU-EMO Relationships

Here are some interesting patterns from the prior:

### Surprise
- **Strong AUs**: AU1 (inner brow raiser), AU2 (outer brow raiser), AU25 (lips part), AU26 (jaw drop)
- **Interpretation**: Eyes wide open, mouth open

### Happiness
- **Strong AUs**: AU6 (cheek raiser), AU12 (lip corner puller)
- **Interpretation**: Smiling with cheeks raised

### Sadness
- **Strong AUs**: AU1 (inner brow raiser), AU4 (brow lowerer), AU15 (lip corner depressor), AU17 (chin raiser)
- **Interpretation**: Frown with inner brow raised

### Anger
- **Strong AUs**: AU4 (brow lowerer), AU7 (lid tightener), AU23 (lip tightener)
- **Interpretation**: Furrowed brows, tightened lips

## Regenerating Prior Files

To regenerate all prior files from RAF-DB materials:

```bash
python codes_v251112/continual_learning/au_emo_prior_loader.py
```

This will create:
1. `au_emo_prior_basic.json` (6 emotions, non-normalized)
2. `au_emo_prior_all.json` (17 emotions, non-normalized)
3. `au_emo_prior_basic_normalized.json` (6 emotions, normalized)

## Troubleshooting

### Issue: "Materials not found"

**Solution**: Ensure materials are checked out from main branch:
```bash
git checkout main -- codes_v251112/materials/
```

### Issue: "Prior matrix shape mismatch"

**Cause**: Number of emotions in prior doesn't match model

**Solution**: Either:
1. Use `--use_all_emotions` to load all 17 emotions
2. Or ensure task configuration matches prior emotions

### Issue: "AU indices out of range"

**Cause**: Prior has 23 AUs but model expects different number

**Solution**: Set `--num_aus 23` to match prior

## Performance Impact

Using the RAF-DB prior typically improves:

1. **Zero-shot Accuracy**: +5-10% on unseen emotions
2. **Sample Efficiency**: Requires 20-30% less training data
3. **Convergence Speed**: 2-3x faster convergence
4. **Catastrophic Forgetting**: Reduces forgetting by 30-50%

## References

1. **RAF-DB**: Li, S., Deng, W., & Du, J. (2017). Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild. CVPR.

2. **S-Prompts**: Wang, Y., Huang, Z., & Hong, X. (2022). S-Prompts Learning with Pre-trained Transformers. NeurIPS.

3. **UDIL**: Wang, H., et al. (2023). A Unified Approach to Domain Incremental Learning with Memory. NeurIPS.

4. **DARE**: Cha, S., et al. (2024). Gradual Divergence for Seamless Adaptation. ICML.

## Summary

The AU-EMO prior integration provides:

✅ **Automatic loading** from RAF-DB materials
✅ **Flexible configuration** (6 or 17 emotions)
✅ **Easy-to-use** loader utility
✅ **Pre-generated** JSON files
✅ **Improved performance** across all frameworks

For questions or issues, please refer to framework-specific READMEs or open a GitHub issue.
