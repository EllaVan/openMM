"""
Domain Incremental Learning with Multimodal Prompts - Main Script

Integrates S-Prompts, UDIL, and DARE methodologies for multimodal emotion recognition.

Example usage:
    python domain_prompt_main.py \
        --data_dir ../../output/mosei_features \
        --task_sequence custom \
        --num_epochs 10 \
        --use_adaptive_weighting \
        --use_contrastive \
        --save_dir ../../checkpoints/domain_prompts
"""

import argparse
import torch
import torch.optim as optim
from pathlib import Path
import sys
import json

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent))

from domain_prompt_trainer import PromptedMultimodalNetwork, DomainPromptTrainer
from continual_learning.domain_splitter import DomainSplitter, create_predefined_task_sequence
from continual_learning.metrics import ContinualLearningMetrics
from hyper_fusion.dataloader import load_mosei_data


def parse_args():
    parser = argparse.ArgumentParser(
        description='Domain Incremental Learning with Prompts'
    )

    # Data
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='MOSEI')

    # Task Configuration
    parser.add_argument('--task_sequence', type=str, default='custom')
    parser.add_argument('--exclude_neutral', action='store_true')

    # Model Architecture
    parser.add_argument('--text_dim', type=int, default=768)
    parser.add_argument('--audio_dim', type=int, default=768)
    parser.add_argument('--video_dim', type=int, default=768)
    parser.add_argument('--num_aus', type=int, default=23)
    parser.add_argument('--prompt_length', type=int, default=5)

    # Training
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)

    # Domain Incremental Settings
    parser.add_argument('--use_adaptive_weighting', action='store_true', default=True)
    parser.add_argument('--use_contrastive', action='store_true', default=True)
    parser.add_argument('--use_alignment', action='store_true', default=True)
    parser.add_argument('--num_prototypes', type=int, default=10)

    # Output
    parser.add_argument('--save_dir', type=str, default='../../checkpoints/domain_prompts')
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
    print("DOMAIN INCREMENTAL LEARNING WITH MULTIMODAL PROMPTS")
    print("="*80)
    print("Integrating: S-Prompts + UDIL + DARE")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {args.device}")

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
    print(f"Number of tasks/domains: {len(tasks)}")
    for task in tasks:
        print(f"  {task}")

    splitter = DomainSplitter(dataset, exclude_neutral=args.exclude_neutral)

    # Determine number of emotions
    num_emotions = len(set([cls for task in tasks for cls in task.seen_classes + task.unseen_classes]))

    # Create model
    print("\n" + "-"*80)
    print("Creating Model...")
    model = PromptedMultimodalNetwork(
        text_input_dim=args.text_dim,
        audio_input_dim=args.audio_dim,
        video_input_dim=args.video_dim,
        num_aus=args.num_aus,
        num_emotions=num_emotions,
        prompt_length=args.prompt_length,
        device=args.device
    )

    print(f"Model created")

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Create trainer
    print("\n" + "-"*80)
    print("Creating Trainer...")
    trainer = DomainPromptTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        use_adaptive_weighting=args.use_adaptive_weighting,
        use_contrastive=args.use_contrastive,
        use_alignment=args.use_alignment,
        num_prototypes_per_domain=args.num_prototypes,
        save_dir=str(save_dir)
    )

    print(f"Trainer initialized")
    print(f"  Adaptive weighting: {args.use_adaptive_weighting}")
    print(f"  Contrastive learning: {args.use_contrastive}")
    print(f"  Feature alignment: {args.use_alignment}")

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
    print("STARTING DOMAIN INCREMENTAL TRAINING")
    print("="*80)

    for task_id, task_config in enumerate(tasks):
        domain_id = f"domain_{task_id}"

        print(f"\n{'='*80}")
        print(f"DOMAIN {task_id}: {task_config.task_name}")
        print(f"  Seen classes: {task_config.seen_classes}")
        print(f"  Unseen classes: {task_config.unseen_classes}")
        print(f"{'='*80}")

        # Create dataloaders
        seen_loader, _ = splitter.create_task_dataloaders(
            task_config,
            batch_size=args.batch_size
        )

        # Train domain
        task_stats = trainer.train_domain(
            domain_id=domain_id,
            domain_name=task_config.task_name,
            train_loader=seen_loader,
            num_epochs=args.num_epochs,
            is_first_domain=(task_id == 0)
        )

        # Evaluate on all previous domains
        print(f"\nEvaluating on all domains...")
        for eval_task_id in range(task_id + 1):
            eval_config = tasks[eval_task_id]
            eval_domain_id = f"domain_{eval_task_id}"

            eval_loader, _ = splitter.create_task_dataloaders(
                eval_config,
                batch_size=args.batch_size,
                shuffle=False
            )

            eval_stats = trainer._evaluate(eval_loader, eval_domain_id)

            # Update metrics (placeholder - need predictions)
            print(f"  Domain {eval_task_id}: acc={eval_stats['accuracy']:.4f}")

        # Print cumulative metrics
        print(f"\nAfter Domain {task_id}:")
        if args.use_adaptive_weighting and task_id > 0:
            weights = trainer.loss_weighting.get_weights()
            print(f"  Adaptive weights: {weights}")

    # Save final results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    # Save training stats
    with open(save_dir / 'training_stats.json', 'w') as f:
        json.dump(trainer.training_stats, f, indent=2, default=str)

    if args.use_adaptive_weighting:
        weight_stats = trainer.loss_weighting.get_weight_stats()
        with open(save_dir / 'adaptive_weights.json', 'w') as f:
            json.dump({k: v.tolist() for k, v in weight_stats.items()}, f, indent=2)

    print(f"\n{'='*80}")
    print("DOMAIN INCREMENTAL TRAINING COMPLETED!")
    print(f"Results saved to: {save_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
