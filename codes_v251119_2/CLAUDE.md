# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a zero-shot continual learning framework for emotion recognition that uses AU (Action Units) - Emotion relationships. The system implements a novel two-stage training approach:

1. **Stage 1 (Seen Training)**: Trains backbone networks on seen emotion samples with EWC (Elastic Weight Consolidation) to prevent catastrophic forgetting
2. **Stage 2 (Unseen Zero-shot)**: Uses EM (Expectation-Maximization) iterations to generate classifier weights for unseen emotions through AU-EMO relationships

### Core Architecture

- **Multi-modal encoding**: Text (RoBERTa), Audio (HuBERT), Video (ViT-Base) features
- **Hypergraph fusion**: Combines multi-modal features using hypergraph neural networks
- **AU prediction**: Predicts 23 Action Units from fused features
- **Zero-shot expansion**: Uses graph convolutional networks to generate unseen emotion classifiers
- **Beta prior management**: Bayesian updating of P(AU|EMO) distributions

## Common Development Tasks

### Running Training

The main training script is [`main_zeroshot.py`](main_zeroshot.py):

```bash
# Basic training
python main_zeroshot.py

# Training with specific config
python main_zeroshot.py --config config/train_config.yaml
```

### Feature Extraction

For dataset preprocessing and feature extraction:

```bash
# IEMOCAP dataset
python dataset/preprocess_iemocap.py

# Extract features from datasets
python dataset/extract_features.py
```

### Demo and Testing

```bash
# Zero-shot demo
python demo/demo_zeroshot.py

# Test incremental emotions
python demo/test_incremental_emotions.py

# KL divergence testing
python demo/test_kl_divergence.py
```

## Configuration System

### Main Configuration Files

- [`config/train_config.yaml`](config/train_config.yaml): Primary training configuration
- [`config/continual_learning.yaml`](config/continual_learning.yaml): Alternative continual learning config
- [`config/tasks.json`](config/tasks.json): Task definitions with seen/unseen emotions

### Key Configuration Parameters

**Training Parameters**:
- `stage1_epochs`: Training epochs for seen emotions (default: 10-20)
- `em_iterations`: EM iterations for zero-shot (default: 20)
- `epochs_per_em`: Training epochs per EM iteration (default: 10)
- `convergence_threshold`: Agreement threshold for EM convergence (default: 0.95)

**Model Architecture**:
- `num_aus`: Number of Action Units (default: 20)
- `encoder_hidden_dim`: Encoder hidden dimension (default: 256)
- `hypergraph_hidden_dim`: Hypergraph fusion dimension (default: 256)
- `num_hyperedges`: Number of hyperedges (default: 64)

**Beta Prior**:
- `pseudo_count`: Beta distribution pseudo count controlling prior strength (default: 230)

## Project Structure

### Core Components

- [`core/au_emotion_network.py`](core/au_emotion_network.py): Main network architecture with multi-modal encoding and AU prediction
- [`core/zeroshot_expander.py`](core/zeroshot_expander.py): Graph convolutional network for zero-shot weight generation
- [`core/beta_au_emo_prior.py`](core/beta_au_emo_prior.py): Bayesian prior management for AU-EMO relationships
- [`core/zeroshot_utils.py`](core/zeroshot_utils.py): Utility functions for zero-shot operations
- [`core/learnable_matrix.py`](core/learnable_matrix.py): Learnable AU-EMO relationship matrix
- [`core/ewc.py`](core/ewc.py): Elastic Weight Consolidation implementation

### Training System

- [`training/two_stage_trainer.py`](training/two_stage_trainer.py): Main trainer implementing the two-stage training paradigm
- [`training/trainer.py`](training/trainer.py): Base trainer utilities

### Data Handling

- [`data/dataloader.py`](data/dataloader.py): Data loaders with seen/unseen separation
- [`dataset/extract_features.py`](dataset/extract_features.py): Feature extraction utilities
- [`dataset/preprocess_iemocap.py`](dataset/preprocess_iemocap.py): IEMOCAP dataset preprocessing

### Fusion Components

- [`fusion/hypergraph_fusion.py`](fusion/hypergraph_fusion.py): Hypergraph-based multi-modal fusion

## Materials and Dependencies

### Required Materials

- [`materials/au_emo_prior.json`](materials/au_emo_prior.json): AU-EMO prior relationship matrix
- [`materials/au_embedding.pt`](materials/au_embedding.pt): Pre-trained AU semantic embeddings
- [`materials/AU_action.txt`](materials/AU_action.txt): AU action definitions

### Data Format

The system expects pre-extracted features in the format described in [`data/instruct.md`](data/instruct.md):

- Each sample contains text_features (768-dim), audio_features (768-dim), video_features (768-dim)
- Emotions are mapped to standardized labels (happy, sad, surprise, disgust, anger, fear, joy)
- Data is organized by dataset and emotion in pickle files

## Task Configuration System

Tasks are defined in [`config/tasks.json`](config/tasks.json) with:
- **seen_emotions**: Emotions available for training in each task
- **unseen_emotions**: New emotions encountered zero-shot
- **Cross-dataset setup**: Tasks can span different datasets (MOSEI, MELD, IEMOCAP)

### Example Task Structure
```json
{
  "task_id": 0,
  "seen_emotions": {"happy": 0, "sad": 1},
  "unseen_emotions": {"surprise": 2, "disgust": 3}
}
```

## Output and Checkpoints

### Model Checkpoints
- Location: `output/zeroshot_continual/`
- `task{id}_final.pt`: Complete model state after each task
- `task{id}_classifier_weights.pt`: Classifier weights for seen+unseen emotions
- `task{id}_beta_prior.npz`: Updated beta prior parameters
- `final_model.pt`: Final model with all tasks

### Training Logs
- Location: `output/logs/`
- Timestamped log files with detailed training progress
- Real-time console output during training

## Development Guidelines

### Modifying Training Pipeline
1. Changes to stage 1 (seen training) should update [`TwoStageTrainer.stage1_train()`](training/two_stage_trainer.py)
2. EM iteration changes require updating both E-step and M-step in [`TwoStageTrainer.stage2_em_training()`](training/two_stage_trainer.py)
3. Architecture changes should be reflected in both [`AUEmotionNetwork`](core/au_emotion_network.py) and model configuration

### Adding New Datasets
1. Follow the feature extraction format in [`data/instruct.md`](data/instruct.md)
2. Add dataset-specific preprocessing in [`dataset/`](dataset/) directory
3. Update task configuration in [`config/tasks.json`](config/tasks.json)

### AU-EMO Relationship Management
- Prior matrix modifications should update [`materials/au_emo_prior.json`](materials/au_emo_prior.json)
- AU embedding changes require updating [`materials/au_embedding.pt`](materials/au_embedding.pt)
- Beta prior hyperparameters are in [`config/train_config.yaml`](config/train_config.yaml) under `beta_prior`

## Performance Considerations

### Memory Optimization
- Reduce `batch_size` in config for GPU memory constraints
- Decrease `num_hyperedges` or hidden dimensions for smaller models
- Use gradient checkpointing for very large models

### Training Speed
- Increase `num_workers` in data configuration for faster loading
- Adjust `save_frequency` to balance I/O overhead
- Use mixed precision training if implemented

### Convergence Issues
- Increase `em_iterations` if EM doesn't converge
- Adjust `convergence_threshold` for stricter/looser convergence
- Modify `zeroshot_lr` for stable EM updates
- Tune `pseudo_count` for appropriate prior strength