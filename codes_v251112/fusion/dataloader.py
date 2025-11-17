"""
样本级别超图融合数据加载器 - 支持 Seen/Unseen Emotions
特征格式：每个样本的每个模态特征为 [768] 维度（无时序）
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import random
import numpy as np


class SampleEmotionDataset(Dataset):
    """
    样本级别多情感数据集

    每个样本包含:
    - audio_features: [audio_dim] (e.g., [768])
    - text_features: [text_dim] (e.g., [768])
    - video_features: [video_dim] (e.g., [768])
    - label: int (原始标签)
    - mapped_label: int (重新映射后的标签)
    - emotion: str
    - is_seen: bool
    """

    def __init__(self, data: List[Dict]):
        """
        Args:
            data: 数据列表，每个元素已包含 mapped_label
        """
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """返回单个样本"""
        return self.data[idx]

    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        emotion_counts = {}
        seen_counts = 0
        unseen_counts = 0

        for item in self.data:
            emotion = item.get('emotion', 'unknown')
            is_seen = item.get('is_seen', True)

            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

            if is_seen:
                seen_counts += 1
            else:
                unseen_counts += 1

        return {
            'size': len(self.data),
            'emotion_counts': emotion_counts,
            'seen_counts': seen_counts,
            'unseen_counts': unseen_counts
        }


def sample_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    样本级别 collate 函数

    Args:
        batch: 样本列表

    Returns:
        batched_data: {
            'audio_features': [batch_size, audio_dim],
            'text_features': [batch_size, text_dim],
            'video_features': [batch_size, video_dim],
            'labels': [batch_size],
            'original_labels': [batch_size],
            'is_seen': [batch_size]
        }
    """
    batch_size = len(batch)

    # 堆叠特征
    audio_features = torch.stack([item['audio_features'] for item in batch])
    text_features = torch.stack([item['text_features'] for item in batch])
    video_features = torch.stack([item['video_features'] for item in batch])

    # 标签
    labels = torch.tensor([item['mapped_label'] for item in batch], dtype=torch.long)
    original_labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    is_seen = torch.tensor([item.get('is_seen', True) for item in batch], dtype=torch.bool)

    return {
        'audio_features': audio_features,
        'text_features': text_features,
        'video_features': video_features,
        'labels': labels,
        'original_labels': original_labels,
        'is_seen': is_seen
    }


def load_pkl_file(file_path: str) -> List[Dict]:
    """加载 pkl 文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 如果是字典，转换为列表
    if isinstance(data, dict):
        data = list(data.values()) if len(data) > 0 else []

    return data


def load_emotion_data(
    data_dir: str,
    dataset_name: str,
    emotion: str,
    label_id: int,
    is_seen: bool = True,
    train_ratio: float = 0.7,
    merge_train_dev: bool = True,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    加载单个 emotion 的数据

    Returns:
        train_data, test_data
    """
    dataset_name = dataset_name.upper()

    if dataset_name == 'MOSEI':
        filename = f"MOSEI{emotion}label{label_id}.pkl"
        file_path = os.path.join(data_dir, filename)
        all_data = load_pkl_file(file_path)

        # 添加 emotion 和 is_seen 信息
        for item in all_data:
            # item['emotion'] = emotion
            item['is_seen'] = is_seen

        # 划分数据
        indices = list(range(len(all_data)))
        random.shuffle(indices)

        train_size = int(len(all_data) * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_data = [all_data[i] for i in train_indices]
        test_data = [all_data[i] for i in test_indices]

    elif dataset_name == 'MELD':
        train_file = f"MELD_train{emotion}label{label_id}.pkl"
        dev_file = f"MELD_dev{emotion}label{label_id}.pkl"
        test_file = f"MELD_test{emotion}label{label_id}.pkl"

        train_data = load_pkl_file(os.path.join(data_dir, train_file))
        dev_data = load_pkl_file(os.path.join(data_dir, dev_file))
        test_data = load_pkl_file(os.path.join(data_dir, test_file))

        # 添加 emotion 和 is_seen 信息
        for item in train_data + dev_data + test_data:
            # item['emotion'] = emotion
            item['is_seen'] = is_seen

        # 合并 train 和 dev
        if merge_train_dev:
            train_data = train_data + dev_data

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return train_data, test_data


def create_emotion_dataloaders(cfg) -> Dict[str, DataLoader]:
    """
    创建支持 seen/unseen emotions 的 DataLoader

    重要：seen 和 unseen emotions 都会重新映射标签
    - seen: 0, 1, 2, ...
    - unseen: len(seen), len(seen)+1, ...

    Args:
        cfg: Hydra DictConfig 对象

    Returns:
        dataloaders: {
            'train': DataLoader (所有数据的训练集),
            'test': DataLoader (所有数据的测试集)
        }
    """
    # 获取配置
    data_dir = cfg.dataset.data_dir
    dataset_name = cfg.dataset.name
    seen_emotions = dict(cfg.dataset.seen_emotions)
    unseen_emotions = dict(cfg.dataset.unseen_emotions) if cfg.dataset.unseen_emotions else {}
    train_ratio = cfg.dataset.train_ratio
    merge_train_dev = cfg.dataset.merge_train_dev
    seed = cfg.system.random_seed

    batch_size = cfg.dataloader.batch_size
    num_workers = cfg.dataloader.num_workers
    shuffle_train = cfg.dataloader.shuffle_train
    pin_memory = cfg.dataloader.pin_memory

    print(f"\n{'='*70}")
    print(f"加载数据集: {dataset_name}")
    print(f"数据目录: {data_dir}")
    print(f"{'='*70}")

    # 创建标签映射: seen emotions -> 0, 1, 2, ..., unseen emotions -> len(seen), len(seen)+1, ...
    emotion_label_map = {}

    # 映射 seen emotions
    for new_label, (emotion, original_label) in enumerate(seen_emotions.items()):
        emotion_label_map[original_label] = new_label

    # 映射 unseen emotions
    unseen_start_label = len(seen_emotions)
    for new_label, (emotion, original_label) in enumerate(unseen_emotions.items()):
        emotion_label_map[original_label] = unseen_start_label + new_label

    print(f"\n标签映射: {emotion_label_map}")
    print(f"  Seen emotions: {list(seen_emotions.keys())} -> {list(range(len(seen_emotions)))}")
    if unseen_emotions:
        print(f"  Unseen emotions: {list(unseen_emotions.keys())} -> {list(range(unseen_start_label, unseen_start_label + len(unseen_emotions)))}")

    # 加载所有数据
    train_all_data = []
    test_all_data = []

    # 1. 加载 seen emotions
    print(f"\n加载 Seen Emotions: {list(seen_emotions.keys())}")
    for emotion, label_id in seen_emotions.items():
        print(f"  加载 {emotion} (原始label={label_id}, 新label={emotion_label_map[label_id]})...")
        train_data, test_data = load_emotion_data(
            data_dir=data_dir,
            dataset_name=dataset_name,
            emotion=emotion,
            label_id=label_id,
            is_seen=True,
            train_ratio=train_ratio,
            merge_train_dev=merge_train_dev,
            seed=seed
        )

        # 重新映射标签
        for item in train_data + test_data:
            item['mapped_label'] = emotion_label_map[label_id]

        train_all_data.extend(train_data)
        test_all_data.extend(test_data)
        print(f"    训练集: {len(train_data)} 样本, 测试集: {len(test_data)} 样本")

    # 2. 加载 unseen emotions
    if unseen_emotions:
        print(f"\n加载 Unseen Emotions: {list(unseen_emotions.keys())}")
        for emotion, label_id in unseen_emotions.items():
            print(f"  加载 {emotion} (原始label={label_id}, 新label={emotion_label_map[label_id]})...")
            train_data, test_data = load_emotion_data(
                data_dir=data_dir,
                dataset_name=dataset_name,
                emotion=emotion,
                label_id=label_id,
                is_seen=False,
                train_ratio=train_ratio,
                merge_train_dev=merge_train_dev,
                seed=seed
            )

            # 重新映射标签
            for item in train_data + test_data:
                item['mapped_label'] = emotion_label_map[label_id]

            train_all_data.extend(train_data)
            test_all_data.extend(test_data)
            print(f"    训练集: {len(train_data)} 样本, 测试集: {len(test_data)} 样本")

    # 创建 Dataset
    train_dataset = SampleEmotionDataset(train_all_data)
    test_dataset = SampleEmotionDataset(test_all_data)

    print(f"\n【数据集统计】")
    print(f"  训练集总样本数: {len(train_dataset)}")
    print(f"  测试集总样本数: {len(test_dataset)}")

    train_stats = train_dataset.get_stats()
    print(f"  训练集情感分布: {train_stats['emotion_counts']}")
    print(f"  Seen样本数: {train_stats['seen_counts']}, Unseen样本数: {train_stats['unseen_counts']}")

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=sample_collate_fn,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=sample_collate_fn,
        pin_memory=pin_memory
    )

    dataloaders = {
        'train': train_loader,
        'test': test_loader
    }

    print(f"{'='*70}\n")

    return dataloaders
