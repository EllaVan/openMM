"""
Blackbox Learnable Continual Learning - Main Execution Script

Complete Task 0 to Task T training with end-to-end learnable AU-EMO matrix.

Example usage:
    python blackbox_main.py \
        --data_dir ../../output/mosei_features \
        --au_prior_path ../example_au_emo_prior.json \
        --task_sequence custom \
        --num_epochs 10 \
        --save_dir ../../checkpoints/blackbox
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import sys
import json

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from learnable_matrix import LearnableAUEMOMatrix, load_au_emo_prior
from gradient_trainer import GradientTrainerBlackbox
from continual_learning.au_emotion_network import AUEmotionNetwork
from continual_learning.domain_splitter import DomainSplitter, create_predefined_task_sequence
from continual_learning.metrics import ContinualLearningMetrics
from continual_learning.consistency_checker import ConsistencyStrategy
from fusion.dataloader import load_mosei_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Blackbox Learnable Continual Learning'
    )

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing preprocessed features')
    parser.add_argument('--dataset', type=str, default='MOSEI',
                       choices=['MOSEI', 'MELD'])

    # AU-EMO Prior
    parser.add_argument('--au_prior_path', type=str, required=True,
                       help='Path to AU-EMO prior JSON file')
    parser.add_argument('--num_aus', type=int, default=23)
    parser.add_argument('--prior_strength', type=float, default=0.1,
                       help='Matrix regularization strength (KL to prior)')

    # Task Configuration
    parser.add_argument('--task_sequence', type=str, default='custom',
                       choices=['demo', 'full', 'custom'])
    parser.add_argument('--task_config_path', type=str, default=None)
    parser.add_argument('--exclude_neutral', action='store_true')

    # Model Architecture
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=768)
    parser.add_argument('--video_dim', type=int, default=768)
    parser.add_argument('--encoder_hidden_dim', type=int, default=256)
    parser.add_argument('--num_hyperedges', type=int, default=64)
    parser.add_argument('--num_conv_layers', type=int, default=2)

    # Training
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs per task')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for network')
    parser.add_argument('--matrix_lr', type=float, default=1e-3,
                       help='Learning rate for matrix (can be different)')
    parser.add_argument('--num_workers', type=int, default=4)

    # Continual Learning
    parser.add_argument('--use_ewc', action='store_true', default=True)
    parser.add_argument('--ewc_lambda', type=float, default=1000.0)
    parser.add_argument('--matrix_reg_lambda', type=float, default=0.1,
                       help='Matrix regularization strength')
    parser.add_argument('--consistency_strategy', type=str, default='majority',
                       choices=['all_agree', 'majority', 'weighted_vote',
                               'entropy_threshold', 'combined'])
    parser.add_argument('--min_confidence', type=float, default=0.8)

    # Loss Weights
    parser.add_argument('--seen_loss_weight', type=float, default=1.0)
    parser.add_argument('--unseen_loss_weight', type=float, default=0.3,
                       help='Lower weight for unseen (pseudo-labels)')

    # Output
    parser.add_argument('--save_dir', type=str,
                       default='../../checkpoints/blackbox')
    parser.add_argument('--log_interval', type=int, default=10)

    # Device
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

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
    print("BLACKBOX LEARNABLE CONTINUAL LEARNING")
    print("="*80)
    print("Framework: End-to-End Learnable AU-EMO Matrix")
    print(f"Dataset: {args.dataset}")
    print(f"Save directory: {args.save_dir}")
    print(f"Device: {args.device}")

    # ===== 1. Load AU-EMO Prior =====
    print("\n" + "-"*80)
    print("Loading AU-EMO Prior...")
    print("-"*80)

    prior_matrix, au_names, emotion_names = load_au_emo_prior(args.au_prior_path)
    num_emotions = len(emotion_names)

    print(f"Prior matrix shape: {prior_matrix.shape}")
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

    # ===== 4. Create Learnable AU-EMO Matrix =====
    print("\n" + "-"*80)
    print("Creating Learnable AU-EMO Matrix...")
    print("-"*80)

    au_emo_matrix = LearnableAUEMOMatrix(
        num_aus=args.num_aus,
        num_emotions=num_emotions,
        prior_p_au_given_emo=prior_matrix,
        prior_strength=args.prior_strength,
        device=args.device
    )

    print(f"Matrix initialized as nn.Parameter")
    print(f"Prior regularization strength: {args.prior_strength}")
    print("\nInitial P(EMO|AU) statistics:")
    print(f"  Mean probability: {au_emo_matrix.get_probability_matrix().mean().item():.4f}")
    print(f"  KL from prior: {au_emo_matrix.get_statistics()['kl_from_prior']:.6f}")

    # ===== 5. Create Model =====
    print("\n" + "-"*80)
    print("Creating Model...")
    print("-"*80)

    model = AUEmotionNetwork(
        text_input_dim=args.text_dim,
        audio_input_dim=args.audio_dim,
        video_input_dim=args.video_dim,
        num_aus=args.num_aus,
        num_emotions=num_emotions,
        au_emo_prior=None,  # We use external learnable matrix
        encoder_hidden_dim=args.encoder_hidden_dim,
        num_hyperedges=args.num_hyperedges,
        num_conv_layers=args.num_conv_layers,
        device=args.device
    )

    # Replace model's internal matrix with our learnable matrix
    model.au_emo_matrix = au_emo_matrix

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    matrix_params = au_emo_matrix.matrix_logits.numel()

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Matrix parameters: {matrix_params} ({100*matrix_params/trainable_params:.2f}%)")

    # ===== 6. Create Optimizer =====
    # Separate learning rates for network and matrix
    optimizer = optim.Adam([
        {'params': [p for n, p in model.named_parameters() if 'matrix_logits' not in n],
         'lr': args.lr},
        {'params': [au_emo_matrix.matrix_logits],
         'lr': args.matrix_lr}
    ])

    print(f"\nOptimizer: Adam")
    print(f"  Network LR: {args.lr}")
    print(f"  Matrix LR: {args.matrix_lr}")

    # ===== 7. Create Gradient Trainer =====
    print("\n" + "-"*80)
    print("Creating Gradient Trainer...")
    print("-"*80)

    strategy_map = {
        'all_agree': ConsistencyStrategy.ALL_AGREE,
        'majority': ConsistencyStrategy.MAJORITY,
        'weighted_vote': ConsistencyStrategy.WEIGHTED_VOTE,
        'entropy_threshold': ConsistencyStrategy.ENTROPY_THRESHOLD,
        'combined': ConsistencyStrategy.COMBINED
    }

    trainer = GradientTrainerBlackbox(
        model=model,
        au_emo_matrix=au_emo_matrix,
        optimizer=optimizer,
        device=args.device,
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        matrix_reg_lambda=args.matrix_reg_lambda,
        consistency_strategy=strategy_map[args.consistency_strategy],
        min_confidence=args.min_confidence,
        seen_loss_weight=args.seen_loss_weight,
        unseen_loss_weight=args.unseen_loss_weight,
        save_dir=str(save_dir)
    )

    print(f"Gradient Trainer initialized")
    print(f"  EWC: {args.use_ewc} (lambda: {args.ewc_lambda})")
    print(f"  Matrix regularization: {args.matrix_reg_lambda}")
    print(f"  Consistency: {args.consistency_strategy}")
    print(f"  Loss weights: seen={args.seen_loss_weight}, unseen={args.unseen_loss_weight}")

    # ===== 8. Create Metrics Tracker =====
    metrics = ContinualLearningMetrics(
        num_tasks=len(tasks),
        num_classes_per_task={
            task.task_id: len(task.seen_classes) + len(task.unseen_classes)
            for task in tasks
        }
    )

    # ===== 9. Training Loop =====
    print("\n" + "="*80)
    print("STARTING BLACKBOX GRADIENT DESCENT TRAINING")
    print("="*80)

    for task_config in tasks:
        print(f"\n{'='*80}")
        print(f"TASK {task_config.task_id}: {task_config.task_name}")
        print(f"  Seen classes: {task_config.seen_classes}")
        print(f"  Unseen classes: {task_config.unseen_classes}")
        print(f"{'='*80}")

        # Create dataloaders
        seen_loader, unseen_loader = splitter.create_task_dataloaders(
            task_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        # Train task with gradient descent
        task_stats = trainer.train_task(
            task_id=task_config.task_id,
            task_name=task_config.task_name,
            seen_loader=seen_loader,
            unseen_loader=unseen_loader,
            num_epochs=args.num_epochs,
            log_interval=args.log_interval
        )

        # Evaluate on all previous tasks
        print(f"\nEvaluating on all tasks...")
        for eval_task_id in range(task_config.task_id + 1):
            eval_task_config = tasks[eval_task_id]

            eval_seen_loader, _ = splitter.create_task_dataloaders(
                eval_task_config,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                shuffle=False
            )

            eval_results = trainer._evaluate_seen(eval_seen_loader)

            metrics.update(
                task_trained=task_config.task_id,
                task_eval=eval_task_id,
                predictions=eval_results['predictions'],
                labels=eval_results['labels']
            )

            print(f"  Task {eval_task_id}: acc={eval_results['accuracy']:.4f}")

        # Print intermediate metrics
        print(f"\nMetrics after Task {task_config.task_id}:")
        print(f"  Average Accuracy: {metrics.average_accuracy(task_config.task_id):.4f}")
        if task_config.task_id > 0:
            print(f"  Average Forgetting: {metrics.average_forgetting(task_config.task_id):.4f}")

        # Print matrix statistics
        print(f"\nAU-EMO Matrix Statistics:")
        matrix_stats = au_emo_matrix.get_statistics()
        for key, value in matrix_stats.items():
            print(f"  {key}: {value:.4f}")

    # ===== 10. Save Final Results =====
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save final model and matrix
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

    # Visualize final AU-EMO matrix
    print("\nFinal P(EMO|AU) Matrix:")
    print(au_emo_matrix.visualize_matrix(
        au_names=au_names,
        emotion_names=emotion_names,
        show_logits=False
    ))

    # Save matrix visualization
    with open(results_dir / 'final_matrix.txt', 'w') as f:
        f.write("Probability Matrix:\n")
        f.write(au_emo_matrix.visualize_matrix(
            au_names=au_names,
            emotion_names=emotion_names,
            show_logits=False
        ))
        f.write("\n\nLogits Matrix:\n")
        f.write(au_emo_matrix.visualize_matrix(
            au_names=au_names,
            emotion_names=emotion_names,
            show_logits=True
        ))

    # Estimate P(AU|EMO) from learned P(EMO|AU)
    p_au_given_emo = au_emo_matrix.get_p_au_given_emo_estimate()
    print("\nEstimated P(AU|EMO) (reverse Bayes):")
    print(f"  Mean: {p_au_given_emo.mean().item():.4f}")

    # Print final summary
    print("\n" + trainer.get_training_summary())
    print("\n" + metrics.get_summary())

    # Save summaries
    with open(save_dir / 'training_summary.txt', 'w') as f:
        f.write("BLACKBOX LEARNABLE CONTINUAL LEARNING\n")
        f.write("="*80 + "\n\n")
        f.write(trainer.get_training_summary())
        f.write("\n\n")
        f.write(metrics.get_summary())
        f.write("\n\n")
        f.write("Final P(EMO|AU) Matrix:\n")
        f.write(au_emo_matrix.visualize_matrix(
            au_names=au_names,
            emotion_names=emotion_names,
            show_logits=False
        ))

    print(f"\n{'='*80}")
    print("BLACKBOX TRAINING COMPLETED!")
    print(f"Results saved to: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
