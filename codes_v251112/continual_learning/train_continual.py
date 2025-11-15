"""
Main Training Script for Continual Learning

Example usage:
    python continual_learning/train_continual.py \
        --data_dir ./output/mosei_features \
        --au_prior_path ./continual_learning/example_au_emo_prior.json \
        --task_sequence custom \
        --num_epochs 10 \
        --batch_size 32 \
        --save_dir ./checkpoints/continual
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from continual_learning import (
    AUEmotionNetwork,
    ContinualLearningTrainer,
    DomainSplitter,
    ContinualLearningMetrics,
    create_predefined_task_sequence,
    load_au_emo_prior,
    ConsistencyStrategy
)
from hyper_fusion.dataloader import load_mosei_data


def parse_args():
    parser = argparse.ArgumentParser(description='Continual Learning Training')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing preprocessed features')
    parser.add_argument('--dataset', type=str, default='MOSEI',
                       choices=['MOSEI', 'MELD'],
                       help='Dataset name')

    # AU-EMO Prior
    parser.add_argument('--au_prior_path', type=str, required=True,
                       help='Path to AU-EMO prior JSON file')
    parser.add_argument('--num_aus', type=int, default=23,
                       help='Number of Action Units')

    # Task Configuration
    parser.add_argument('--task_sequence', type=str, default='custom',
                       choices=['demo', 'full', 'custom'],
                       help='Predefined task sequence')
    parser.add_argument('--task_config_path', type=str, default=None,
                       help='Path to custom task configuration JSON')
    parser.add_argument('--exclude_neutral', action='store_true',
                       help='Exclude neutral emotion samples')

    # Model Architecture
    parser.add_argument('--text_dim', type=int, default=768,
                       help='Text feature dimension')
    parser.add_argument('--audio_dim', type=int, default=768,
                       help='Audio feature dimension')
    parser.add_argument('--video_dim', type=int, default=768,
                       help='Video feature dimension')
    parser.add_argument('--encoder_hidden_dim', type=int, default=256,
                       help='Encoder hidden dimension')
    parser.add_argument('--num_hyperedges', type=int, default=64,
                       help='Number of hyperedges')
    parser.add_argument('--num_conv_layers', type=int, default=2,
                       help='Number of hypergraph convolution layers')

    # Training
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of dataloader workers')

    # Continual Learning
    parser.add_argument('--use_ewc', action='store_true', default=True,
                       help='Use EWC for anti-forgetting')
    parser.add_argument('--ewc_lambda', type=float, default=1000.0,
                       help='EWC regularization weight')
    parser.add_argument('--ewc_type', type=str, default='online',
                       choices=['standard', 'online', 'selective'],
                       help='Type of EWC')

    # Consistency Checking
    parser.add_argument('--consistency_strategy', type=str, default='majority',
                       choices=['all_agree', 'majority', 'weighted_vote', 'entropy_threshold', 'combined'],
                       help='Consistency checking strategy')
    parser.add_argument('--min_confidence', type=float, default=0.8,
                       help='Minimum confidence for unseen updates')

    # AU-EMO Matrix
    parser.add_argument('--prior_strength', type=float, default=100.0,
                       help='AU-EMO prior strength')
    parser.add_argument('--seen_update_weight', type=float, default=10.0,
                       help='Weight for seen class updates')
    parser.add_argument('--unseen_update_weight', type=float, default=1.0,
                       help='Weight for unseen class updates')
    parser.add_argument('--au_emo_regularization', type=float, default=0.01,
                       help='AU-EMO regularization towards prior')

    # Output
    parser.add_argument('--save_dir', type=str, default='./checkpoints/continual',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='Logging interval (batches)')
    parser.add_argument('--eval_interval', type=int, default=1,
                       help='Evaluation interval (epochs)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "="*80)
    print("CONTINUAL LEARNING TRAINING")
    print("="*80)
    print(f"Dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print(f"Save directory: {args.save_dir}")
    print(f"Device: {args.device}")

    # ===== 1. Load AU-EMO Prior =====
    print("\n" + "-"*80)
    print("Loading AU-EMO Prior...")
    print("-"*80)

    prior_matrix, au_names, emotion_names = load_au_emo_prior(args.au_prior_path)
    num_emotions = len(emotion_names)

    print(f"Loaded prior matrix: {prior_matrix.shape}")
    print(f"AUs: {len(au_names)}")
    print(f"Emotions: {emotion_names}")

    # ===== 2. Load Dataset =====
    print("\n" + "-"*80)
    print("Loading Dataset...")
    print("-"*80)

    if args.dataset == 'MOSEI':
        dataset = load_mosei_data(
            data_dir=args.data_dir,
            emotion='all'
        )
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    print(f"Dataset size: {len(dataset)}")

    # ===== 3. Create Task Sequence =====
    print("\n" + "-"*80)
    print("Creating Task Sequence...")
    print("-"*80)

    if args.task_config_path:
        tasks = DomainSplitter.load_task_configs(args.task_config_path)
    else:
        tasks = create_predefined_task_sequence(args.task_sequence, args.dataset)

    print(f"Number of tasks: {len(tasks)}")
    for task in tasks:
        print(f"  {task}")

    # Save task configurations
    splitter = DomainSplitter(dataset, exclude_neutral=args.exclude_neutral)
    splitter.save_task_configs(tasks, str(save_dir / 'task_configs.json'))

    # ===== 4. Create Model =====
    print("\n" + "-"*80)
    print("Creating Model...")
    print("-"*80)

    model = AUEmotionNetwork(
        text_input_dim=args.text_dim,
        audio_input_dim=args.audio_dim,
        video_input_dim=args.video_dim,
        num_aus=args.num_aus,
        num_emotions=num_emotions,
        au_emo_prior=prior_matrix,
        encoder_hidden_dim=args.encoder_hidden_dim,
        num_hyperedges=args.num_hyperedges,
        num_conv_layers=args.num_conv_layers,
        au_emo_prior_strength=args.prior_strength,
        device=args.device
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ===== 5. Create Optimizer =====
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ===== 6. Create Trainer =====
    print("\n" + "-"*80)
    print("Creating Trainer...")
    print("-"*80)

    # Map strategy name to enum
    strategy_map = {
        'all_agree': ConsistencyStrategy.ALL_AGREE,
        'majority': ConsistencyStrategy.MAJORITY,
        'weighted_vote': ConsistencyStrategy.WEIGHTED_VOTE,
        'entropy_threshold': ConsistencyStrategy.ENTROPY_THRESHOLD,
        'combined': ConsistencyStrategy.COMBINED
    }

    trainer = ContinualLearningTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        ewc_type=args.ewc_type,
        consistency_strategy=strategy_map[args.consistency_strategy],
        min_confidence=args.min_confidence,
        seen_update_weight=args.seen_update_weight,
        unseen_update_weight=args.unseen_update_weight,
        au_emo_regularization=args.au_emo_regularization,
        save_dir=str(save_dir)
    )

    print(f"EWC: {args.use_ewc} (type: {args.ewc_type}, lambda: {args.ewc_lambda})")
    print(f"Consistency: {args.consistency_strategy} (min_conf: {args.min_confidence})")

    # ===== 7. Create Metrics Tracker =====
    metrics = ContinualLearningMetrics(
        num_tasks=len(tasks),
        num_classes_per_task={
            task.task_id: len(task.seen_classes) + len(task.unseen_classes)
            for task in tasks
        }
    )

    # ===== 8. Training Loop =====
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    for task_config in tasks:
        print(f"\n{'='*80}")
        print(f"TASK {task_config.task_id}: {task_config.task_name}")
        print(f"{'='*80}")

        # Create dataloaders
        seen_loader, unseen_loader = splitter.create_task_dataloaders(
            task_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Train task
        task_stats = trainer.train_task(
            task_id=task_config.task_id,
            task_name=task_config.task_name,
            seen_loader=seen_loader,
            unseen_loader=unseen_loader,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval,
            evaluate_interval=args.eval_interval
        )

        # Evaluate on all tasks seen so far
        print(f"\nEvaluating on all tasks...")
        for eval_task_id in range(task_config.task_id + 1):
            eval_task_config = tasks[eval_task_id]

            # Create evaluation dataloader (only seen classes for fair comparison)
            eval_seen_loader, _ = splitter.create_task_dataloaders(
                eval_task_config,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False
            )

            # Evaluate
            eval_results = trainer.evaluate(
                eval_seen_loader,
                phase=f'task_{eval_task_id}'
            )

            # Update metrics
            if 'accuracy' in eval_results:
                metrics.update(
                    task_trained=task_config.task_id,
                    task_eval=eval_task_id,
                    predictions=eval_results['predictions'],
                    labels=eval_results['labels']
                )

        # Print intermediate metrics
        print(f"\nMetrics after Task {task_config.task_id}:")
        print(f"  Average Accuracy: {metrics.average_accuracy(task_config.task_id):.4f}")
        if task_config.task_id > 0:
            print(f"  Average Forgetting: {metrics.average_forgetting(task_config.task_id):.4f}")

    # ===== 9. Save Final Model and Results =====
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save final model
    trainer.save_final_model('final_model.pt')

    # Save metrics
    metrics.save_metrics(str(save_dir / 'metrics.json'))

    # Plot results
    results_dir = save_dir / 'results'
    results_dir.mkdir(exist_ok=True)

    metrics.plot_performance_matrix(
        save_path=str(results_dir / 'performance_matrix.png'),
        task_names=[t.task_name for t in tasks]
    )

    metrics.plot_learning_curves(
        save_path=str(results_dir / 'learning_curves.png'),
        task_names=[t.task_name for t in tasks]
    )

    # Print final summary
    print(trainer.get_training_summary())
    print(metrics.get_summary())

    # Save summaries as text
    with open(save_dir / 'training_summary.txt', 'w') as f:
        f.write(trainer.get_training_summary())
        f.write("\n\n")
        f.write(metrics.get_summary())

    print(f"\n{'='*80}")
    print("TRAINING COMPLETED!")
    print(f"Results saved to: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
