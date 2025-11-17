"""
Domain Splitting Utilities for Continual Learning

Splits datasets into tasks with seen and unseen emotion classes.
Supports both single-dataset splitting and multi-dataset scenarios.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from collections import defaultdict


# Emotion mappings
EMOTION_NAMES = {
    0: 'happy',
    1: 'sad',
    2: 'angry',
    3: 'surprise',
    4: 'disgust',
    5: 'fear',
    6: 'neutral'
}

EMOTION_IDS = {v: k for k, v in EMOTION_NAMES.items()}


class TaskConfig:
    """
    Configuration for a single task

    Attributes:
    -----------
    task_id : int
        Task identifier
    task_name : str
        Task name (e.g., "MOSEI_Task1")
    dataset_name : str
        Dataset name ('MOSEI' or 'MELD')
    seen_classes : List[int]
        Emotion IDs for seen classes
    unseen_classes : List[int]
        Emotion IDs for unseen classes
    data_split : float or tuple
        If float: fraction of data to use
        If tuple: (start_idx, end_idx) for dataset slicing
    """

    def __init__(
        self,
        task_id: int,
        task_name: str,
        dataset_name: str,
        seen_classes: List[int],
        unseen_classes: List[int],
        data_split: Optional[float] = None
    ):
        self.task_id = task_id
        self.task_name = task_name
        self.dataset_name = dataset_name
        self.seen_classes = sorted(seen_classes)
        self.unseen_classes = sorted(unseen_classes)
        self.data_split = data_split

    def __repr__(self):
        seen_names = [EMOTION_NAMES[i] for i in self.seen_classes]
        unseen_names = [EMOTION_NAMES[i] for i in self.unseen_classes]
        return (f"TaskConfig(id={self.task_id}, name='{self.task_name}', "
                f"seen={seen_names}, unseen={unseen_names})")

    def to_dict(self):
        """Convert to dictionary for JSON serialization"""
        return {
            'task_id': self.task_id,
            'task_name': self.task_name,
            'dataset_name': self.dataset_name,
            'seen_classes': self.seen_classes,
            'unseen_classes': self.unseen_classes,
            'seen_class_names': [EMOTION_NAMES[i] for i in self.seen_classes],
            'unseen_class_names': [EMOTION_NAMES[i] for i in self.unseen_classes],
            'data_split': self.data_split
        }

    @classmethod
    def from_dict(cls, config_dict):
        """Create from dictionary"""
        return cls(
            task_id=config_dict['task_id'],
            task_name=config_dict['task_name'],
            dataset_name=config_dict['dataset_name'],
            seen_classes=config_dict['seen_classes'],
            unseen_classes=config_dict['unseen_classes'],
            data_split=config_dict.get('data_split', None)
        )


class DomainSplitter:
    """
    Domain Splitter for Continual Learning

    Splits dataset(s) into multiple tasks with seen/unseen classes.

    Parameters:
    -----------
    dataset : Dataset
        The complete dataset
    exclude_neutral : bool
        Whether to exclude neutral emotion
    random_seed : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        dataset: Dataset,
        exclude_neutral: bool = True,
        random_seed: int = 42
    ):
        self.dataset = dataset
        self.exclude_neutral = exclude_neutral
        self.random_seed = random_seed

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Analyze dataset
        self._analyze_dataset()

    def _analyze_dataset(self):
        """Analyze dataset to get class distribution"""
        print("Analyzing dataset...")

        # Count samples per class
        self.class_counts = defaultdict(int)
        self.class_indices = defaultdict(list)

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            label = sample['label'] if isinstance(sample, dict) else sample[1]

            if isinstance(label, torch.Tensor):
                label = label.item()

            # Skip neutral if excluded
            if self.exclude_neutral and label == EMOTION_IDS['neutral']:
                continue

            self.class_counts[label] += 1
            self.class_indices[label].append(idx)

        print("\nClass distribution:")
        for label_id in sorted(self.class_counts.keys()):
            emotion_name = EMOTION_NAMES[label_id]
            count = self.class_counts[label_id]
            print(f"  {emotion_name}: {count} samples")

    def create_tasks_by_strategy(
        self,
        strategy: str = 'small_unseen',
        num_tasks: int = 3,
        seen_classes_base: Optional[List[int]] = None
    ) -> List[TaskConfig]:
        """
        Create task configurations based on strategy

        Strategies:
        -----------
        'small_unseen': Small-sample classes as unseen
        'incremental': Incrementally add new seen classes
        'disjoint': Completely different seen/unseen per task
        'overlap': Overlapping seen classes, different unseen

        Args:
            strategy: Splitting strategy
            num_tasks: Number of tasks to create
            seen_classes_base: Base seen classes to use across tasks

        Returns:
            List of TaskConfig objects
        """
        available_classes = sorted(self.class_counts.keys())

        if strategy == 'small_unseen':
            return self._strategy_small_unseen(num_tasks, seen_classes_base)
        elif strategy == 'incremental':
            return self._strategy_incremental(num_tasks)
        elif strategy == 'disjoint':
            return self._strategy_disjoint(num_tasks)
        elif strategy == 'overlap':
            return self._strategy_overlap(num_tasks, seen_classes_base)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _strategy_small_unseen(
        self,
        num_tasks: int,
        seen_classes_base: Optional[List[int]] = None
    ) -> List[TaskConfig]:
        """
        Strategy: Use small-sample classes as unseen

        This follows your preference: larger sample classes are seen,
        smaller sample classes are unseen.
        """
        # Sort classes by sample count (descending)
        classes_by_count = sorted(
            self.class_counts.keys(),
            key=lambda x: self.class_counts[x],
            reverse=True
        )

        if seen_classes_base is None:
            # Use top 2 largest classes as base seen classes
            seen_classes_base = classes_by_count[:2]

        # Remaining classes are candidates for unseen
        unseen_candidates = [c for c in classes_by_count if c not in seen_classes_base]

        # Split unseen candidates across tasks
        unseen_per_task = []
        for i in range(num_tasks):
            start_idx = i * len(unseen_candidates) // num_tasks
            end_idx = (i + 1) * len(unseen_candidates) // num_tasks
            unseen_per_task.append(unseen_candidates[start_idx:end_idx])

        # Create task configs
        tasks = []
        for task_id in range(num_tasks):
            # Vary seen classes per task
            if task_id == 0:
                seen_classes = seen_classes_base
            elif task_id % 2 == 1:
                # Odd tasks: only one seen class
                seen_classes = [seen_classes_base[0]]
            else:
                # Even tasks: both seen classes
                seen_classes = seen_classes_base

            task_config = TaskConfig(
                task_id=task_id,
                task_name=f"Task{task_id}_SmallUnseen",
                dataset_name="MOSEI",
                seen_classes=seen_classes,
                unseen_classes=unseen_per_task[task_id],
                data_split=1.0 / num_tasks  # Split data evenly
            )
            tasks.append(task_config)

        return tasks

    def _strategy_incremental(self, num_tasks: int) -> List[TaskConfig]:
        """
        Strategy: Incrementally add classes as seen

        Task 1: 1 seen, rest unseen
        Task 2: 2 seen, rest unseen
        Task 3: 3 seen, rest unseen
        ...
        """
        classes_by_count = sorted(
            self.class_counts.keys(),
            key=lambda x: self.class_counts[x],
            reverse=True
        )

        tasks = []
        for task_id in range(num_tasks):
            num_seen = min(task_id + 1, len(classes_by_count) - 1)
            seen_classes = classes_by_count[:num_seen]
            unseen_classes = classes_by_count[num_seen:]

            task_config = TaskConfig(
                task_id=task_id,
                task_name=f"Task{task_id}_Incremental",
                dataset_name="MOSEI",
                seen_classes=seen_classes,
                unseen_classes=unseen_classes
            )
            tasks.append(task_config)

        return tasks

    def _strategy_disjoint(self, num_tasks: int) -> List[TaskConfig]:
        """
        Strategy: Completely different classes per task

        Divide all classes into groups, each task gets one group as seen,
        rest as unseen.
        """
        all_classes = sorted(self.class_counts.keys())

        # Divide classes into groups
        classes_per_task = len(all_classes) // num_tasks

        tasks = []
        for task_id in range(num_tasks):
            start_idx = task_id * classes_per_task
            end_idx = (task_id + 1) * classes_per_task if task_id < num_tasks - 1 else len(all_classes)

            seen_classes = all_classes[start_idx:end_idx]
            unseen_classes = [c for c in all_classes if c not in seen_classes]

            task_config = TaskConfig(
                task_id=task_id,
                task_name=f"Task{task_id}_Disjoint",
                dataset_name="MOSEI",
                seen_classes=seen_classes,
                unseen_classes=unseen_classes
            )
            tasks.append(task_config)

        return tasks

    def _strategy_overlap(
        self,
        num_tasks: int,
        seen_classes_base: Optional[List[int]] = None
    ) -> List[TaskConfig]:
        """
        Strategy: Overlapping seen classes, different unseen

        All tasks share the same seen classes (e.g., happy, sad),
        but have different unseen classes.
        """
        classes_by_count = sorted(
            self.class_counts.keys(),
            key=lambda x: self.class_counts[x],
            reverse=True
        )

        if seen_classes_base is None:
            seen_classes_base = classes_by_count[:2]

        unseen_candidates = [c for c in classes_by_count if c not in seen_classes_base]

        tasks = []
        for task_id in range(num_tasks):
            # Rotate unseen classes
            unseen_classes = [
                unseen_candidates[(task_id + i) % len(unseen_candidates)]
                for i in range(len(unseen_candidates) // num_tasks + 1)
            ]

            task_config = TaskConfig(
                task_id=task_id,
                task_name=f"Task{task_id}_Overlap",
                dataset_name="MOSEI",
                seen_classes=seen_classes_base,
                unseen_classes=unseen_classes[:2]  # Max 2 unseen per task
            )
            tasks.append(task_config)

        return tasks

    def create_task_dataloaders(
        self,
        task_config: TaskConfig,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader]:
        """
        Create DataLoaders for a task

        Returns:
            seen_loader: DataLoader for seen class samples
            unseen_loader: DataLoader for unseen class samples
        """
        # Get indices for seen and unseen classes
        seen_indices = []
        for class_id in task_config.seen_classes:
            seen_indices.extend(self.class_indices[class_id])

        unseen_indices = []
        for class_id in task_config.unseen_classes:
            unseen_indices.extend(self.class_indices[class_id])

        # Apply data split if specified
        if task_config.data_split is not None:
            if isinstance(task_config.data_split, float):
                # Use fraction of data
                np.random.shuffle(seen_indices)
                np.random.shuffle(unseen_indices)

                seen_size = int(len(seen_indices) * task_config.data_split)
                unseen_size = int(len(unseen_indices) * task_config.data_split)

                seen_indices = seen_indices[:seen_size]
                unseen_indices = unseen_indices[:unseen_size]

        print(f"\nTask {task_config.task_id}: {task_config.task_name}")
        print(f"  Seen classes: {[EMOTION_NAMES[i] for i in task_config.seen_classes]}")
        print(f"  Seen samples: {len(seen_indices)}")
        print(f"  Unseen classes: {[EMOTION_NAMES[i] for i in task_config.unseen_classes]}")
        print(f"  Unseen samples: {len(unseen_indices)}")

        # Create subsets
        seen_subset = Subset(self.dataset, seen_indices)
        unseen_subset = Subset(self.dataset, unseen_indices)

        # Create DataLoaders
        seen_loader = DataLoader(
            seen_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )

        unseen_loader = DataLoader(
            unseen_subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        ) if len(unseen_indices) > 0 else None

        return seen_loader, unseen_loader

    def save_task_configs(self, task_configs: List[TaskConfig], filepath: str):
        """Save task configurations to JSON file"""
        config_dicts = [task.to_dict() for task in task_configs]

        output = {
            'num_tasks': len(task_configs),
            'exclude_neutral': self.exclude_neutral,
            'random_seed': self.random_seed,
            'class_distribution': {
                EMOTION_NAMES[k]: v for k, v in self.class_counts.items()
            },
            'tasks': config_dicts
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nTask configurations saved to {filepath}")

    @staticmethod
    def load_task_configs(filepath: str) -> List[TaskConfig]:
        """Load task configurations from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        task_configs = [TaskConfig.from_dict(task_dict) for task_dict in data['tasks']]

        print(f"Loaded {len(task_configs)} task configurations from {filepath}")
        return task_configs


def create_predefined_task_sequence(
    sequence_name: str,
    dataset_name: str = "MOSEI"
) -> List[TaskConfig]:
    """
    Create predefined task sequences

    Predefined sequences:
    ---------------------
    'demo': 3 tasks for quick testing
    'full': 6 tasks covering all emotions
    'custom': User's example from conversation

    Args:
        sequence_name: Name of predefined sequence
        dataset_name: Dataset name

    Returns:
        List of TaskConfig objects
    """
    if sequence_name == 'demo':
        # Quick demo: 3 tasks
        tasks = [
            TaskConfig(
                task_id=0,
                task_name=f"{dataset_name}_Task0",
                dataset_name=dataset_name,
                seen_classes=[0, 1],  # happy, sad
                unseen_classes=[2]     # angry
            ),
            TaskConfig(
                task_id=1,
                task_name=f"{dataset_name}_Task1",
                dataset_name=dataset_name,
                seen_classes=[0],      # happy
                unseen_classes=[4, 5]  # disgust, fear
            ),
            TaskConfig(
                task_id=2,
                task_name=f"{dataset_name}_Task2",
                dataset_name=dataset_name,
                seen_classes=[0, 1],  # happy, sad
                unseen_classes=[3]     # surprise
            )
        ]

    elif sequence_name == 'full':
        # Full sequence: all emotions
        tasks = [
            TaskConfig(0, f"{dataset_name}_Task0", dataset_name, [0, 1], [2]),
            TaskConfig(1, f"{dataset_name}_Task1", dataset_name, [0, 1], [3]),
            TaskConfig(2, f"{dataset_name}_Task2", dataset_name, [0], [4]),
            TaskConfig(3, f"{dataset_name}_Task3", dataset_name, [0, 1], [5]),
            TaskConfig(4, f"{dataset_name}_Task4", dataset_name, [0], [2, 3]),
            TaskConfig(5, f"{dataset_name}_Task5", dataset_name, [0, 1], [4, 5])
        ]

    elif sequence_name == 'custom':
        # Your example from conversation
        tasks = [
            TaskConfig(0, f"{dataset_name}_Task0", dataset_name, [0, 1], [2]),      # seen=[happy,sad], unseen=[angry]
            TaskConfig(1, f"{dataset_name}_Task1", dataset_name, [0], [4, 5]),      # seen=[happy], unseen=[disgust,fear]
            TaskConfig(2, f"{dataset_name}_Task2", dataset_name, [0, 1], [3])       # seen=[happy,sad], unseen=[surprise]
        ]

    else:
        raise ValueError(f"Unknown sequence name: {sequence_name}")

    return tasks


if __name__ == "__main__":
    print("Domain Splitter module ready!")
    print("\nPredefined task sequences:")
    for seq_name in ['demo', 'full', 'custom']:
        print(f"\n{seq_name}:")
        tasks = create_predefined_task_sequence(seq_name)
        for task in tasks:
            print(f"  {task}")
