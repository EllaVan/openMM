"""
DILLB-Style Domain Incremental Learning - Main Execution Script

Complete Task 0 to Task T training with DILLB-inspired multi-head architecture.

Example usage:
    python dillb_main.py \
        --data_dir ../../output/mosei_features \
        --au_prior_path ../example_au_emo_prior.json \
        --task_sequence custom \
        --num_epochs 10 \
        --use_distillation \
        --freeze_backbone_after_task0 \
        --save_dir ../../checkpoints/dillb
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from multi_head_network import MultiHeadMultimodalNetwork
from dillb_trainer import DILLBTrainer
from continual_learning.domain_splitter import DomainSplitter, create_predefined_task_sequence
from continual_learning.metrics import ContinualLearningMetrics
from continual_learning.consistency_checker import ConsistencyStrategy
from hyper_fusion.dataloader import load_mosei_data


def parse_args():
    parser = argparse.ArgumentParser(description='DILLB Domain Incremental Learning')

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='MOSEI')
    parser.add_argument('--au_prior_path', type=str, required=True)
    parser.add_argument('--num_aus', type=int, default=23)

    # Task Configuration
    parser.add_argument('--task_sequence', type=str, default='custom')
    parser.add_argument('--exclude_neutral', action='store_true')

    # Model
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=768)
    parser.add_argument('--video_dim', type=int, default=768)
    parser.add_argument('--encoder_hidden_dim', type=int, default=256)

    # Training
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=int, default=1e-4)

    # DILLB Settings
    parser.add_argument('--use_distillation', action='store_true', default=True)
    parser.add_argument('--kd_temperature', type=float, default=2.0)
    parser.add_argument('--alpha_kd', type=float, default=0.3)
    parser.add_argument('--freeze_backbone_after_task0', action='store_true')
    parser.add_argument('--global_matrix_weight', type=float, default=0.5)

    # EWC
    parser.add_argument('--use_ewc', action='store_true', default=True)
    parser.add_argument('--ewc_lambda', type=float, default=1000.0)

    # Consistency
    parser.add_argument('--consistency_strategy', type=str, default='majority')
    parser.add_argument('--min_confidence', type=float, default=0.8)

    # Output
    parser.add_argument('--save_dir', type=str, default='../../checkpoints/dillb')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save args
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    print("\n" + "="*80)
    print("DILLB DOMAIN INCREMENTAL LEARNING")
    print("="*80)
    print(f"Multi-Head Architecture + Knowledge Distillation")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")

    # Load prior
    print("\n" + "-"*80)
    print("Loading AU-EMO Prior...")
    with open(args.au_prior_path, 'r') as f:
        prior_data = json.load(f)
    prior_matrix = torch.tensor(prior_data['prior_matrix'], dtype=torch.float32)
    num_emotions = prior_matrix.shape[1]

    # Load dataset
    print("\n" + "-"*80)
    print("Loading Dataset...")
    if args.dataset == 'MOSEI':
        dataset = load_mosei_data(data_dir=args.data_dir, emotion='all')
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    print(f"Dataset size: {len(dataset)}")

    # Create task sequence
    print("\n" + "-"*80)
    print("Creating Task Sequence...")
    tasks = create_predefined_task_sequence(args.task_sequence, args.dataset)
    print(f"Number of tasks: {len(tasks)}")
    for task in tasks:
        print(f"  {task}")

    splitter = DomainSplitter(dataset, exclude_neutral=args.exclude_neutral)

    # Create model
    print("\n" + "-"*80)
    print("Creating Multi-Head Model...")
    model = MultiHeadMultimodalNetwork(
        text_input_dim=args.text_dim,
        audio_input_dim=args.audio_dim,
        video_input_dim=args.video_dim,
        num_aus=args.num_aus,
        num_emotions=num_emotions,
        au_emo_prior=prior_matrix,
        freeze_backbone=False,  # Will freeze after Task 0 if flag set
        global_matrix_weight=args.global_matrix_weight,
        device=args.device
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create trainer
    print("\n" + "-"*80)
    print("Creating DILLB Trainer...")

    strategy_map = {
        'all_agree': ConsistencyStrategy.ALL_AGREE,
        'majority': ConsistencyStrategy.MAJORITY,
        'weighted_vote': ConsistencyStrategy.WEIGHTED_VOTE,
        'entropy_threshold': ConsistencyStrategy.ENTROPY_THRESHOLD,
        'combined': ConsistencyStrategy.COMBINED
    }

    trainer = DILLBTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        use_distillation=args.use_distillation,
        kd_temperature=args.kd_temperature,
        alpha_kd=args.alpha_kd,
        use_ewc=args.use_ewc,
        ewc_lambda=args.ewc_lambda,
        consistency_strategy=strategy_map[args.consistency_strategy],
        min_confidence=args.min_confidence,
        save_dir=str(save_dir)
    )

    print(f"Trainer initialized")
    print(f"  Knowledge Distillation: {args.use_distillation}")
    print(f"  EWC: {args.use_ewc}")
    print(f"  Freeze backbone after Task 0: {args.freeze_backbone_after_task0}")

    # Create metrics
    metrics = ContinualLearningMetrics(
        num_tasks=len(tasks),
        num_classes_per_task={
            task.task_id: len(task.seen_classes) + len(task.unseen_classes)
            for task in tasks
        }
    )

    # Training loop
    print("\n" + "="*80)
    print("STARTING DILLB TRAINING")
    print("="*80)

    for task_config in tasks:
        domain_id = f"task_{task_config.task_id}"

        print(f"\n{'='*80}")
        print(f"TASK {task_config.task_id}: {task_config.task_name}")
        print(f"  Domain ID: {domain_id}")
        print(f"  Seen classes: {task_config.seen_classes}")
        print(f"  Unseen classes: {task_config.unseen_classes}")
        print(f"{'='*80}")

        # Create dataloaders
        seen_loader, unseen_loader = splitter.create_task_dataloaders(
            task_config,
            batch_size=args.batch_size
        )

        # Train task
        freeze_backbone = (task_config.task_id > 0 and args.freeze_backbone_after_task0)

        task_stats = trainer.train_task(
            task_id=task_config.task_id,
            task_name=task_config.task_name,
            domain_id=domain_id,
            seen_loader=seen_loader,
            unseen_loader=unseen_loader,
            num_epochs=args.num_epochs,
            freeze_backbone=freeze_backbone
        )

        # Evaluate on all tasks
        print(f"\nEvaluating on all tasks...")
        all_results = trainer.evaluate_all_tasks(
            task_configs=tasks[:task_config.task_id+1],
            data_splitter=splitter,
            batch_size=args.batch_size
        )

        # Update metrics
        for eval_task_id, results in enumerate(all_results.values()):
            metrics.update(
                task_trained=task_config.task_id,
                task_eval=eval_task_id,
                predictions=results['predictions'],
                labels=results['labels']
            )

        # Print metrics
        print(f"\nMetrics after Task {task_config.task_id}:")
        print(f"  Average Accuracy: {metrics.average_accuracy(task_config.task_id):.4f}")
        if task_config.task_id > 0:
            print(f"  Average Forgetting: {metrics.average_forgetting(task_config.task_id):.4f}")

    # Save final results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    metrics.save_metrics(str(save_dir / 'metrics.json'))

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

    print(f"\n{'='*80}")
    print("DILLB TRAINING COMPLETED!")
    print(f"Results saved to: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
