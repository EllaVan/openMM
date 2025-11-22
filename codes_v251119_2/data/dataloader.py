"""
持续学习数据加载器 - 支持类增量学习的标签重映射

核心功能：
1. 从 task_config.json 加载任务配置
2. 维护全局标签映射（seen类保持原标签，unseen类顺序分配新标签）
3. 返回重映射的标签和 is_seen 标记

标签分配规则：
- Task 0: seen [happy=0, sad=1], unseen [surprise=2, disgust=3]
- Task 1: seen [happy=0, anger=4], unseen [fear=5]
  (happy 保持 0，anger 是新seen类分配 4)
"""

import os
import json
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np


class IncrementalLabelMapper:
    """
    类增量学习标签映射器

    功能：
    1. 维护从原始标签到增量标签的映射
    2. 追踪哪些类是seen，哪些是unseen
    3. 为新出现的类分配递增的标签

    示例：
        Task 0: seen=[happy:0, sad:1], unseen=[surprise:3, disgust:4]
            -> happy->0, sad->1, surprise->2, disgust->3

        Task 1: seen=[happy:0, anger:2], unseen=[fear:5]
            -> happy->0 (保持), anger->4 (新), fear->5 (新)
    """

    def __init__(self):
        self.original_to_incremental = {}  # 原始标签 -> 增量标签
        self.incremental_to_original = {}  # 增量标签 -> 原始标签
        self.next_incremental_label = 0    # 下一个可用的增量标签
        self.seen_labels = set()            # 已出现过的seen类（原始标签）
        self.emotion_name_to_incremental = {}  # 情绪名 -> 增量标签

    def add_task(
        self,
        task_id: int,
        seen_emotions: Dict[str, int],
        unseen_emotions: Dict[str, int]
    ) -> Dict:
        """
        为新任务添加标签映射

        Args:
            task_id: 任务ID
            seen_emotions: {emotion_name: original_label}
            unseen_emotions: {emotion_name: original_label}

        Returns:
            mapping_info: 映射信息字典
        """
        print(f"Task {task_id}: 标签映射")

        seen_mapping = {}
        unseen_mapping = {}

        # 1. 处理 seen emotions
        print(f"\nSeen emotions:")
        for emotion_name, original_label in seen_emotions.items():
            if original_label in self.original_to_incremental:
                # 之前已经出现过，保持原映射
                incremental_label = self.original_to_incremental[original_label]
                print(f"  {emotion_name} (原始={original_label}) -> {incremental_label} [保持]")
            else:
                # 新出现的seen类，分配新标签
                incremental_label = self.next_incremental_label
                self.original_to_incremental[original_label] = incremental_label
                self.incremental_to_original[incremental_label] = original_label
                self.next_incremental_label += 1
                print(f"  {emotion_name} (原始={original_label}) -> {incremental_label} [新分配]")

            self.seen_labels.add(original_label)
            self.emotion_name_to_incremental[emotion_name] = incremental_label
            seen_mapping[emotion_name] = incremental_label

        # 2. 处理 unseen emotions
        print(f"\nUnseen emotions:")
        for emotion_name, original_label in unseen_emotions.items():
            if original_label in self.original_to_incremental:
                # 之前已经出现过（可能在之前任务是unseen），保持原映射
                incremental_label = self.original_to_incremental[original_label]
                print(f"  {emotion_name} (原始={original_label}) -> {incremental_label} [保持]")
            else:
                # 新出现的unseen类，分配新标签
                incremental_label = self.next_incremental_label
                self.original_to_incremental[original_label] = incremental_label
                self.incremental_to_original[incremental_label] = original_label
                self.next_incremental_label += 1
                print(f"  {emotion_name} (原始={original_label}) -> {incremental_label} [新分配]")

            self.emotion_name_to_incremental[emotion_name] = incremental_label
            unseen_mapping[emotion_name] = incremental_label

        print(f"\n当前全局映射: {self.original_to_incremental}")
        print(f"下一个可用标签: {self.next_incremental_label}")

        return {
            'task_id': task_id,
            'seen_mapping': seen_mapping,
            'unseen_mapping': unseen_mapping,
            'global_mapping': dict(self.original_to_incremental),
            'next_label': self.next_incremental_label
        }

    def get_incremental_label(self, original_label: int) -> int:
        """获取增量标签"""
        if original_label not in self.original_to_incremental:
            raise ValueError(f"原始标签 {original_label} 未注册")
        return self.original_to_incremental[original_label]

    def is_seen(self, original_label: int) -> bool:
        """判断是否是seen类"""
        return original_label in self.seen_labels

    def get_num_classes_so_far(self) -> int:
        """获取目前为止出现的总类数"""
        return self.next_incremental_label


class ContinualLearningDataset(Dataset):
    """
    持续学习数据集

    每个样本包含:
    - audio_features: [audio_dim]
    - text_features: [text_dim]
    - video_features: [video_dim]
    - original_label: int (原始标签)
    - label: int (增量标签，重映射后)
    - is_seen: bool (是否是seen类)
    - emotion_name: str
    """

    def __init__(self, data: List[Dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        return self.data[idx]

    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        emotion_counts = {}
        seen_counts = 0
        unseen_counts = 0
        label_counts = {}

        for item in self.data:
            emotion = item['emotion_name']
            is_seen = item['is_seen']
            label = item['label']

            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            label_counts[label] = label_counts.get(label, 0) + 1

            if is_seen:
                seen_counts += 1
            else:
                unseen_counts += 1

        return {
            'total': len(self.data),
            'emotion_counts': emotion_counts,
            'label_counts': label_counts,
            'seen_count': seen_counts,
            'unseen_count': unseen_counts
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate函数

    Returns:
        {
            'text': [batch_size, text_dim],
            'audio': [batch_size, audio_dim],
            'video': [batch_size, video_dim],
            'label': [batch_size],  # 增量标签
            'original_label': [batch_size],  # 原始标签
            'is_seen': [batch_size]  # bool tensor
        }
    """
    # 提取特征
    text = torch.stack([item['text_features'] for item in batch])
    audio = torch.stack([item['audio_features'] for item in batch])
    video = torch.stack([item['video_features'] for item in batch])

    # 标签
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    original_labels = torch.tensor([item['original_label'] for item in batch], dtype=torch.long)
    is_seen = torch.tensor([item['is_seen'] for item in batch], dtype=torch.bool)

    return {
        'text': text,
        'audio': audio,
        'video': video,
        'label': labels,
        'original_label': original_labels,
        'is_seen': is_seen
    }


def load_emotion_pkl(
    data_dir: str,
    dataset_name: str,
    emotion_name: str,
    original_label: int
) -> List[Dict]:
    """
    加载单个情绪的pkl文件

    Args:
        data_dir: 数据目录
        dataset_name: 数据集名称 (MOSEI/MELD)
        emotion_name: 情绪名称
        original_label: 原始标签ID

    Returns:
        数据列表
    """
    dataset_name = dataset_name.upper()

    if dataset_name == 'MOSEI':
        # MOSEI 格式: MOSEIhappylabel0.pkl
        filename = f"{dataset_name}{emotion_name}.pkl"
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            print(f"  警告: 文件不存在 {file_path}")
            return []

        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 如果是字典，转为列表
        if isinstance(data, dict):
            data = list(data.values())

        return data

    elif dataset_name == 'MELD':
        # MELD 格式: MELD_trainhappylabel0.pkl, MELD_devhappylabel0.pkl, MELD_testhappylabel0.pkl
        train_file = f"{dataset_name}_train{emotion_name}.pkl"
        dev_file = f"{dataset_name}_dev{emotion_name}.pkl"
        test_file = f"{dataset_name}_test{emotion_name}.pkl"

        all_data = []
        for filename in [train_file, dev_file, test_file]:
            file_path = os.path.join(data_dir, filename)
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    data = list(data.values())
                all_data.extend(data)

        return all_data

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")


def create_task_dataloaders(
    task_config_path: str,
    task_id: int,
    label_mapper: Optional[IncrementalLabelMapper] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    shuffle_train: bool = True,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, IncrementalLabelMapper, Dict]:
    """
    为指定任务创建数据加载器

    Args:
        task_config_path: 任务配置JSON文件路径
        task_id: 任务ID
        label_mapper: 标签映射器（如果为None则创建新的）
        batch_size: batch大小
        num_workers: 工作进程数
        train_ratio: 训练集比例
        shuffle_train: 是否打乱训练集
        seed: 随机种子

    Returns:
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        label_mapper: 更新后的标签映射器
        task_info: 任务信息字典
    """
    # 设置随机种子
    # np.random.seed(seed)
    # torch.manual_seed(seed)

    # 1. 加载任务配置
    print(f"\n{'='*80}")
    print(f"加载任务配置: {task_config_path}")
    print(f"{'='*80}")

    with open(task_config_path, 'r') as f:
        config = json.load(f)

    if task_id >= len(config['tasks']):
        raise ValueError(f"任务ID {task_id} 超出范围 (总任务数: {len(config['tasks'])})")

    task_cfg = config['tasks'][task_id]

    # 支持每个任务使用不同的数据集和数据目录
    # 优先使用任务级别的配置，否则使用全局默认值
    dataset_name = task_cfg.get('dataset_name', config.get('default_dataset', 'MOSEI'))
    data_dir = task_cfg.get('data_dir', config.get('default_data_dir', '../../output/mosei_features'))

    print(f"\n任务信息:")
    print(f"  Task ID: {task_cfg['task_id']}")
    print(f"  Task Name: {task_cfg['task_name']}")
    print(f"  Description: {task_cfg.get('description', 'N/A')}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Data Dir: {data_dir}")

    # 2. 创建或更新标签映射器
    if label_mapper is None:
        label_mapper = IncrementalLabelMapper()

    mapping_info = label_mapper.add_task(
        task_id=task_id,
        seen_emotions=task_cfg['seen_emotions'],
        unseen_emotions=task_cfg.get('unseen_emotions', {})
    )

    # 3. 加载数据
    print(f"\n{'='*70}")
    print(f"加载数据文件...")
    print(f"{'='*70}")

    all_data = []

    # 3.1 加载seen emotions
    print(f"\n[Seen Emotions]")
    for emotion_name, original_label in task_cfg['seen_emotions'].items():
        incremental_label = label_mapper.get_incremental_label(original_label)

        print(f"  加载 {emotion_name} (原始={original_label}, 增量={incremental_label})...")

        emotion_data = load_emotion_pkl(data_dir, dataset_name, emotion_name, original_label)

        if len(emotion_data) == 0:
            print(f"    警告: 没有数据")
            continue

        # 添加标签信息
        for item in emotion_data:
            item['original_label'] = original_label
            item['label'] = incremental_label
            item['is_seen'] = True
            item['emotion_name'] = emotion_name

        all_data.extend(emotion_data)
        print(f"    加载了 {len(emotion_data)} 个样本")

    # 3.2 加载unseen emotions
    unseen_emotions = task_cfg.get('unseen_emotions', {})
    if unseen_emotions:
        print(f"\n[Unseen Emotions]")
        for emotion_name, original_label in unseen_emotions.items():
            incremental_label = label_mapper.get_incremental_label(original_label)

            print(f"  加载 {emotion_name} (原始={original_label}, 增量={incremental_label})...")

            emotion_data = load_emotion_pkl(data_dir, dataset_name, emotion_name, original_label)

            if len(emotion_data) == 0:
                print(f"    警告: 没有数据")
                continue

            # 添加标签信息
            for item in emotion_data:
                item['original_label'] = original_label
                item['label'] = incremental_label
                item['is_seen'] = False
                item['emotion_name'] = emotion_name

            all_data.extend(emotion_data)
            print(f"    加载了 {len(emotion_data)} 个样本")

    # 4. 划分训练集和测试集
    print(f"\n{'='*70}")
    print(f"划分训练集和测试集 (比例: {train_ratio:.2f})")
    print(f"{'='*70}")

    np.random.shuffle(all_data)

    train_size = int(len(all_data) * train_ratio)
    train_data = all_data[:train_size]
    test_data = all_data[train_size:]

    # 5. 创建Dataset
    train_dataset = ContinualLearningDataset(train_data)
    test_dataset = ContinualLearningDataset(test_data)

    # 打印统计信息
    train_stats = train_dataset.get_stats()
    test_stats = test_dataset.get_stats()

    print(f"\n训练集统计:")
    print(f"  总样本数: {train_stats['total']}")
    print(f"  Seen样本: {train_stats['seen_count']}")
    print(f"  Unseen样本: {train_stats['unseen_count']}")
    print(f"  情绪分布: {train_stats['emotion_counts']}")
    print(f"  标签分布: {train_stats['label_counts']}")

    print(f"\n测试集统计:")
    print(f"  总样本数: {test_stats['total']}")
    print(f"  Seen样本: {test_stats['seen_count']}")
    print(f"  Unseen样本: {test_stats['unseen_count']}")
    print(f"  情绪分布: {test_stats['emotion_counts']}")
    print(f"  标签分布: {test_stats['label_counts']}")

    # 6. 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    print(f"\nDataLoader创建完成:")
    print(f"  训练批次数: {len(train_loader)}")
    print(f"  测试批次数: {len(test_loader)}")
    print(f"  Batch大小: {batch_size}")
    print(f"{'='*80}\n")

    # 7. 任务信息
    task_info = {
        'task_id': task_id,
        'task_name': task_cfg['task_name'],
        'dataset_name': dataset_name,
        'data_dir': data_dir,
        'seen_emotions': task_cfg['seen_emotions'],
        'unseen_emotions': task_cfg.get('unseen_emotions', {}),
        'mapping_info': mapping_info,
        'train_stats': train_stats,
        'test_stats': test_stats,
        'num_classes_so_far': label_mapper.get_num_classes_so_far()
    }

    return train_loader, test_loader, label_mapper, task_info


def load_all_tasks(
    task_config_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    seed: int = 42
) -> List[Tuple[DataLoader, DataLoader, Dict]]:
    """
    加载所有任务的数据

    Args:
        task_config_path: 任务配置文件路径
        batch_size: batch大小
        num_workers: 工作进程数
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        tasks_data: [(train_loader, test_loader, task_info), ...]
    """
    # 读取任务配置
    with open(task_config_path, 'r') as f:
        config = json.load(f)

    num_tasks = len(config['tasks'])

    print(f"\n{'#'*80}")
    print(f"# 加载全部 {num_tasks} 个任务")
    print(f"{'#'*80}\n")

    tasks_data = []
    label_mapper = IncrementalLabelMapper()

    for task_id in range(num_tasks):
        train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
            task_config_path=task_config_path,
            task_id=task_id,
            label_mapper=label_mapper,
            batch_size=batch_size,
            num_workers=num_workers,
            train_ratio=train_ratio,
            seed=seed
        )

        tasks_data.append((train_loader, test_loader, task_info))

    print(f"\n{'#'*80}")
    print(f"# 全部任务加载完成")
    print(f"# 总类数: {label_mapper.get_num_classes_so_far()}")
    print(f"# 全局标签映射: {label_mapper.original_to_incremental}")
    print(f"{'#'*80}\n")

    return tasks_data


def create_task_dataloaders_separated(
    task_config_path: str,
    task_id: int,
    label_mapper: Optional[IncrementalLabelMapper] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.8,
    shuffle_train: bool = True,
    seed: int = 42
) -> Tuple[Dict[str, DataLoader], Dict[str, DataLoader], IncrementalLabelMapper, Dict]:
    """
    为指定任务创建seen/unseen分离的数据加载器

    Args:
        task_config_path: 任务配置JSON文件路径
        task_id: 任务ID
        label_mapper: 标签映射器（如果为None则创建新的）
        batch_size: batch大小
        num_workers: 工作进程数
        train_ratio: 训练集比例
        shuffle_train: 是否打乱训练集
        seed: 随机种子

    Returns:
        train_loaders: {'seen': DataLoader, 'unseen': DataLoader}
        test_loaders: {'seen': DataLoader, 'unseen': DataLoader}
        label_mapper: 更新后的标签映射器
        task_info: 任务信息字典
    """
    # 1. 加载任务配置
    print(f"加载任务配置: {task_config_path}")

    with open(task_config_path, 'r') as f:
        config = json.load(f)

    if task_id >= len(config['tasks']):
        raise ValueError(f"任务ID {task_id} 超出范围 (总任务数: {len(config['tasks'])})")

    task_cfg = config['tasks'][task_id]

    # 支持每个任务使用不同的数据集和数据目录
    dataset_name = task_cfg.get('dataset_name', config.get('default_dataset', 'MOSEI'))
    data_dir = task_cfg.get('data_dir', config.get('default_data_dir', '../../output/mosei_features'))

    print(f"\n任务信息:")
    print(f"  Task ID: {task_cfg['task_id']}")
    print(f"  Task Name: {task_cfg['task_name']}")
    print(f"  Description: {task_cfg.get('description', 'N/A')}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Data Dir: {data_dir}")

    # 2. 创建或更新标签映射器
    if label_mapper is None:
        label_mapper = IncrementalLabelMapper()

    mapping_info = label_mapper.add_task(
        task_id=task_id,
        seen_emotions=task_cfg['seen_emotions'],
        unseen_emotions=task_cfg.get('unseen_emotions', {})
    )

    # 3. 分别加载seen和unseen数据
    print(f"加载数据文件（seen/unseen分离）...")

    seen_data = []
    unseen_data = []

    # 3.1 加载seen emotions
    print(f"\n[Seen Emotions]")
    for emotion_name, original_label in task_cfg['seen_emotions'].items():
        incremental_label = label_mapper.get_incremental_label(original_label)

        print(f"  加载 {emotion_name} (原始={original_label}, 增量={incremental_label})...")

        emotion_data = load_emotion_pkl(data_dir, dataset_name, emotion_name, original_label)

        if len(emotion_data) == 0:
            print(f"    警告: 没有数据")
            continue

        # 添加标签信息
        for item in emotion_data:
            item['original_label'] = original_label
            item['label'] = incremental_label
            item['is_seen'] = True
            item['emotion_name'] = emotion_name

        seen_data.extend(emotion_data)
        print(f"    加载了 {len(emotion_data)} 个样本")

    # 3.2 加载unseen emotions
    unseen_emotions = task_cfg.get('unseen_emotions', {})
    if unseen_emotions:
        print(f"\n[Unseen Emotions]")
        for emotion_name, original_label in unseen_emotions.items():
            incremental_label = label_mapper.get_incremental_label(original_label)

            print(f"  加载 {emotion_name} (原始={original_label}, 增量={incremental_label})...")

            emotion_data = load_emotion_pkl(data_dir, dataset_name, emotion_name, original_label)

            if len(emotion_data) == 0:
                print(f"    警告: 没有数据")
                continue

            # 添加标签信息
            for item in emotion_data:
                item['original_label'] = original_label
                item['label'] = incremental_label
                item['is_seen'] = False
                item['emotion_name'] = emotion_name

            unseen_data.extend(emotion_data)
            print(f"    加载了 {len(emotion_data)} 个样本")

    # 4. 分别划分训练集和测试集
    print(f"划分训练集和测试集 (比例: {train_ratio:.2f})")

    # Seen数据划分
    np.random.shuffle(seen_data)
    seen_train_size = int(len(seen_data) * train_ratio)
    seen_train_data = seen_data[:seen_train_size]
    seen_test_data = seen_data[seen_train_size:]

    # Unseen数据划分
    unseen_train_data = []
    unseen_test_data = []
    if len(unseen_data) > 0:
        np.random.shuffle(unseen_data)
        unseen_train_size = int(len(unseen_data) * train_ratio)
        unseen_train_data = unseen_data[:unseen_train_size]
        unseen_test_data = unseen_data[unseen_train_size:]

    # 5. 创建Dataset
    seen_train_dataset = ContinualLearningDataset(seen_train_data)
    seen_test_dataset = ContinualLearningDataset(seen_test_data)

    # 打印统计信息
    seen_train_stats = seen_train_dataset.get_stats()
    seen_test_stats = seen_test_dataset.get_stats()

    print(f"\n[Seen] 训练集统计:")
    print(f"  总样本数: {seen_train_stats['total']}")
    print(f"  情绪分布: {seen_train_stats['emotion_counts']}")

    print(f"\n[Seen] 测试集统计:")
    print(f"  总样本数: {seen_test_stats['total']}")
    print(f"  情绪分布: {seen_test_stats['emotion_counts']}")

    # 6. 创建DataLoader
    seen_train_loader = DataLoader(
        seen_train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    seen_test_loader = DataLoader(
        seen_test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    train_loaders = {'seen': seen_train_loader}
    test_loaders = {'seen': seen_test_loader}

    # Unseen数据加载器（如果有）
    if len(unseen_train_data) > 0:
        unseen_train_dataset = ContinualLearningDataset(unseen_train_data)
        unseen_test_dataset = ContinualLearningDataset(unseen_test_data)

        unseen_train_stats = unseen_train_dataset.get_stats()
        unseen_test_stats = unseen_test_dataset.get_stats()

        print(f"\n[Unseen] 训练集统计:")
        print(f"  总样本数: {unseen_train_stats['total']}")
        print(f"  情绪分布: {unseen_train_stats['emotion_counts']}")

        print(f"\n[Unseen] 测试集统计:")
        print(f"  总样本数: {unseen_test_stats['total']}")
        print(f"  情绪分布: {unseen_test_stats['emotion_counts']}")

        unseen_train_loader = DataLoader(
            unseen_train_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        unseen_test_loader = DataLoader(
            unseen_test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )

        train_loaders['unseen'] = unseen_train_loader
        test_loaders['unseen'] = unseen_test_loader
    else:
        print(f"\n[Unseen] 无unseen数据")

    print(f"\nDataLoader创建完成:")
    print(f"  Seen训练批次数: {len(train_loaders['seen'])}")
    print(f"  Seen测试批次数: {len(test_loaders['seen'])}")
    if 'unseen' in train_loaders:
        print(f"  Unseen训练批次数: {len(train_loaders['unseen'])}")
        print(f"  Unseen测试批次数: {len(test_loaders['unseen'])}")
    print(f"  Batch大小: {batch_size}")

    # 7. 任务信息
    task_info = {
        'task_id': task_id,
        'task_name': task_cfg['task_name'],
        'dataset_name': dataset_name,
        'data_dir': data_dir,
        'seen_emotions': task_cfg['seen_emotions'],
        'unseen_emotions': task_cfg.get('unseen_emotions', {}),
        'mapping_info': mapping_info,
        'seen_train_stats': seen_train_stats,
        'seen_test_stats': seen_test_stats,
        'unseen_train_stats': unseen_train_stats if len(unseen_train_data) > 0 else None,
        'unseen_test_stats': unseen_test_stats if len(unseen_test_data) > 0 else None,
        'num_classes_so_far': label_mapper.get_num_classes_so_far()
    }

    return train_loaders, test_loaders, label_mapper, task_info


if __name__ == "__main__":
    """测试代码"""
    print("="*80)
    print("测试持续学习数据加载器")
    print("="*80)

    # 测试配置文件路径
    task_config_path = "../../../codes_v251119/config/task_config.json"

    # 测试单个任务
    print("\n【测试1: 加载Task 0】")
    train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
        task_config_path=task_config_path,
        task_id=0,
        batch_size=4,
        num_workers=0,
        train_ratio=0.8
    )

    # 查看一个batch
    print("\n查看训练集第一个batch:")
    batch = next(iter(train_loader))
    print(f"  text shape: {batch['text'].shape}")
    print(f"  audio shape: {batch['audio'].shape}")
    print(f"  video shape: {batch['video'].shape}")
    print(f"  label: {batch['label']}")
    print(f"  original_label: {batch['original_label']}")
    print(f"  is_seen: {batch['is_seen']}")

    # 测试Task 1（使用同一个label_mapper）
    print("\n【测试2: 加载Task 1】")
    train_loader1, test_loader1, label_mapper, task_info1 = create_task_dataloaders(
        task_config_path=task_config_path,
        task_id=1,
        label_mapper=label_mapper,  # 传入之前的mapper
        batch_size=4,
        num_workers=0,
        train_ratio=0.8
    )

    # 测试加载所有任务
    print("\n【测试3: 加载所有任务】")
    all_tasks = load_all_tasks(
        task_config_path=task_config_path,
        batch_size=4,
        num_workers=0,
        train_ratio=0.8
    )

    print(f"\n加载了 {len(all_tasks)} 个任务")
    for i, (train_loader, test_loader, task_info) in enumerate(all_tasks):
        print(f"\nTask {i}:")
        print(f"  Name: {task_info['task_name']}")
        print(f"  训练批次: {len(train_loader)}")
        print(f"  测试批次: {len(test_loader)}")
        print(f"  当前总类数: {task_info['num_classes_so_far']}")

    print("\n✓ 测试完成!")
