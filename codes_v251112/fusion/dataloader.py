"""
超图融合数据加载器 - 支持 Padding + Masking + Seen/Unseen Emotions
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import random
import numpy as np
from config_utils import Config


class MultiEmotionDataset(Dataset):
    """
    多情感数据集 - 支持 seen/unseen emotions

    数据格式:
    - MELD: MELD_{split}{emotion}label{label_id}.pkl
    - MOSEI: MOSEI{emotion}label{label_id}.pkl

    每个样本包含:
    - audio_features: [num_frames, audio_dim]
    - text_features: [num_frames, text_dim]
    - video_features: [num_frames, video_dim]
    - label: int (原始标签)
    - mapped_label: int (重新映射后的标签，0 到 num_classes-1)
    - num_frames: int
    - emotion: str
    - is_seen: bool (是否为 seen emotion)
    """

    def __init__(
        self,
        data: List[Dict],
        emotion_label_map: Optional[Dict[int, int]] = None
    ):
        """
        Args:
            data: 数据列表，每个元素为字典
            emotion_label_map: 原始标签到新标签的映射 {原始label: 新label}
                              例如: {0: 0, 1: 1} 将 happy(0) 和 sad(1) 映射到 0 和 1
        """
        self.data = data
        self.emotion_label_map = emotion_label_map or {}

        # 如果提供了映射，重新映射标签
        if self.emotion_label_map:
            for item in self.data:
                original_label = item['label']
                if original_label in self.emotion_label_map:
                    item['mapped_label'] = self.emotion_label_map[original_label]
                else:
                    # unseen emotion，不映射标签
                    item['mapped_label'] = -1  # -1 表示无标签

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict:
        """
        返回单个样本

        Returns:
            sample: {
                'audio_features': [num_frames, audio_dim],
                'text_features': [num_frames, text_dim],
                'video_features': [num_frames, video_dim],
                'label': int (原始标签),
                'mapped_label': int (映射后的标签, -1 表示无标签),
                'num_frames': int,
                'emotion': str,
                'is_seen': bool,
                'sample_id': str (可选)
            }
        """
        return self.data[idx]

    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        num_frames_list = [item['num_frames'] for item in self.data]

        # 统计每种 emotion 的样本数
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
            'avg_frames': np.mean(num_frames_list),
            'median_frames': np.median(num_frames_list),
            'min_frames': np.min(num_frames_list),
            'max_frames': np.max(num_frames_list),
            'std_frames': np.std(num_frames_list),
            'emotion_counts': emotion_counts,
            'seen_counts': seen_counts,
            'unseen_counts': unseen_counts
        }


def padded_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    支持变长序列的 collate 函数
    将批次内的样本填充到相同长度，并生成 mask

    Args:
        batch: 样本列表，每个样本为字典

    Returns:
        batched_data: {
            'audio_features': [batch_size, max_frames, audio_dim],
            'text_features': [batch_size, max_frames, text_dim],
            'video_features': [batch_size, max_frames, video_dim],
            'masks': [batch_size, max_frames],  # True 表示有效帧
            'labels': [batch_size],  # 映射后的标签
            'original_labels': [batch_size],  # 原始标签
            'num_frames': [batch_size],  # 每个样本的实际帧数
            'is_seen': [batch_size]  # 是否为 seen emotion
        }
    """
    batch_size = len(batch)

    # 找到批次内最大帧数
    max_frames = max(item['num_frames'] for item in batch)

    # 获取特征维度
    audio_dim = batch[0]['audio_features'].shape[1]
    text_dim = batch[0]['text_features'].shape[1]
    video_dim = batch[0]['video_features'].shape[1]

    # 初始化填充后的张量
    audio_padded = torch.zeros(batch_size, max_frames, audio_dim)
    text_padded = torch.zeros(batch_size, max_frames, text_dim)
    video_padded = torch.zeros(batch_size, max_frames, video_dim)
    masks = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    labels = torch.tensor([item['mapped_label'] for item in batch], dtype=torch.long)
    original_labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    num_frames_list = torch.tensor([item['num_frames'] for item in batch], dtype=torch.long)
    is_seen = torch.tensor([item.get('is_seen', True) for item in batch], dtype=torch.bool)

    # 填充数据
    for i, item in enumerate(batch):
        num_frames = item['num_frames']

        # 填充特征
        audio_padded[i, :num_frames] = item['audio_features']
        text_padded[i, :num_frames] = item['text_features']
        video_padded[i, :num_frames] = item['video_features']

        # 标记有效帧
        masks[i, :num_frames] = True

    return {
        'audio_features': audio_padded,
        'text_features': text_padded,
        'video_features': video_padded,
        'masks': masks,
        'labels': labels,
        'original_labels': original_labels,
        'num_frames': num_frames_list,
        'is_seen': is_seen
    }


def load_pkl_file(file_path: str) -> List[Dict]:
    """
    加载 pkl 文件

    Args:
        file_path: pkl 文件路径

    Returns:
        data: 数据列表
    """
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

    Args:
        data_dir: 数据目录
        dataset_name: 'MOSEI' 或 'MELD'
        emotion: 情感类型
        label_id: 标签 ID
        is_seen: 是否为 seen emotion
        train_ratio: 训练集比例（MOSEI）
        merge_train_dev: 是否合并 train 和 dev（MELD）
        seed: 随机种子

    Returns:
        train_data, test_data
    """
    dataset_name = dataset_name.upper()

    if dataset_name == 'MOSEI':
        # MOSEI 数据集
        filename = f"MOSEI{emotion}label{label_id}.pkl"
        file_path = os.path.join(data_dir, filename)

        all_data = load_pkl_file(file_path)

        # 添加 emotion 和 is_seen 信息
        for item in all_data:
            item['emotion'] = emotion
            item['is_seen'] = is_seen

        # 划分数据
        random.seed(seed)
        indices = list(range(len(all_data)))
        random.shuffle(indices)

        train_size = int(len(all_data) * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_data = [all_data[i] for i in train_indices]
        test_data = [all_data[i] for i in test_indices]

    elif dataset_name == 'MELD':
        # MELD 数据集
        train_file = f"MELD_train{emotion}label{label_id}.pkl"
        dev_file = f"MELD_dev{emotion}label{label_id}.pkl"
        test_file = f"MELD_test{emotion}label{label_id}.pkl"

        train_data = load_pkl_file(os.path.join(data_dir, train_file))
        dev_data = load_pkl_file(os.path.join(data_dir, dev_file))
        test_data = load_pkl_file(os.path.join(data_dir, test_file))

        # 添加 emotion 和 is_seen 信息
        for item in train_data + dev_data + test_data:
            item['emotion'] = emotion
            item['is_seen'] = is_seen

        # 合并 train 和 dev
        if merge_train_dev:
            train_data = train_data + dev_data

    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    return train_data, test_data


def create_multi_emotion_dataloaders(
    config: Config
) -> Dict[str, DataLoader]:
    """
    创建支持多个 emotions 的 DataLoader

    根据配置文件中的 seen_emotions 和 unseen_emotions 创建数据加载器
    - Seen emotions: 有标签，用于训练和测试
    - Unseen emotions: 可选择性加载，用于无监督学习或评估

    Args:
        config: 配置对象

    Returns:
        dataloaders: {
            'train_seen': DataLoader (seen emotions 的训练数据),
            'test_seen': DataLoader (seen emotions 的测试数据),
            'train_unseen': DataLoader (unseen emotions 的训练数据, 可选),
            'test_unseen': DataLoader (unseen emotions 的测试数据, 可选)
        }
    """
    # 获取配置
    data_dir = config.dataset.get('data_dir')
    dataset_name = config.dataset.get('name')
    seen_emotions = config.get_seen_emotions()
    unseen_emotions = config.get_unseen_emotions()
    train_ratio = config.dataset.get('train_ratio', 0.7)
    merge_train_dev = config.dataset.get('merge_train_dev', True)
    seed = config.system.get('random_seed', 42)

    batch_size = config.dataloader.get('batch_size', 32)
    num_workers = config.dataloader.get('num_workers', 4)
    shuffle_train = config.dataloader.get('shuffle_train', True)
    pin_memory = config.dataloader.get('pin_memory', True)

    print(f"\n{'='*70}")
    print(f"加载数据集: {dataset_name}")
    print(f"数据目录: {data_dir}")
    print(f"{'='*70}")

    # 创建标签映射: 将 seen emotions 的原始标签映射到 0, 1, 2, ...
    emotion_label_map = {}
    for new_label, (emotion, original_label) in enumerate(seen_emotions.items()):
        emotion_label_map[original_label] = new_label

    print(f"\n标签映射: {emotion_label_map}")

    # 1. 加载 seen emotions 数据
    print(f"\n加载 Seen Emotions: {list(seen_emotions.keys())}")
    train_seen_data = []
    test_seen_data = []

    for emotion, label_id in seen_emotions.items():
        print(f"  加载 {emotion} (label_id={label_id})...")
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
        train_seen_data.extend(train_data)
        test_seen_data.extend(test_data)
        print(f"    训练集: {len(train_data)} 样本, 测试集: {len(test_data)} 样本")

    # 创建 seen emotions 的 Dataset
    train_seen_dataset = MultiEmotionDataset(train_seen_data, emotion_label_map)
    test_seen_dataset = MultiEmotionDataset(test_seen_data, emotion_label_map)

    print(f"\n【Seen Emotions 统计】")
    print(f"  训练集总样本数: {len(train_seen_dataset)}")
    print(f"  测试集总样本数: {len(test_seen_dataset)}")

    train_stats = train_seen_dataset.get_stats()
    print(f"\n  训练集帧数统计:")
    print(f"    平均: {train_stats['avg_frames']:.1f}")
    print(f"    中位数: {train_stats['median_frames']:.1f}")
    print(f"    范围: [{train_stats['min_frames']:.0f}, {train_stats['max_frames']:.0f}]")
    print(f"  情感分布: {train_stats['emotion_counts']}")

    # 创建 DataLoader
    dataloaders = {}

    dataloaders['train_seen'] = DataLoader(
        train_seen_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=padded_collate_fn,
        pin_memory=pin_memory
    )

    dataloaders['test_seen'] = DataLoader(
        test_seen_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=padded_collate_fn,
        pin_memory=pin_memory
    )

    # 2. 加载 unseen emotions 数据（可选）
    if unseen_emotions:
        print(f"\n加载 Unseen Emotions: {list(unseen_emotions.keys())}")
        train_unseen_data = []
        test_unseen_data = []

        for emotion, label_id in unseen_emotions.items():
            print(f"  加载 {emotion} (label_id={label_id})...")
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
            train_unseen_data.extend(train_data)
            test_unseen_data.extend(test_data)
            print(f"    训练集: {len(train_data)} 样本, 测试集: {len(test_data)} 样本")

        # 创建 unseen emotions 的 Dataset (不进行标签映射)
        train_unseen_dataset = MultiEmotionDataset(train_unseen_data, {})
        test_unseen_dataset = MultiEmotionDataset(test_unseen_data, {})

        print(f"\n【Unseen Emotions 统计】")
        print(f"  训练集总样本数: {len(train_unseen_dataset)}")
        print(f"  测试集总样本数: {len(test_unseen_dataset)}")

        unseen_stats = train_unseen_dataset.get_stats()
        print(f"  情感分布: {unseen_stats['emotion_counts']}")

        dataloaders['train_unseen'] = DataLoader(
            train_unseen_dataset,
            batch_size=batch_size,
            shuffle=shuffle_train,
            num_workers=num_workers,
            collate_fn=padded_collate_fn,
            pin_memory=pin_memory
        )

        dataloaders['test_unseen'] = DataLoader(
            test_unseen_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=padded_collate_fn,
            pin_memory=pin_memory
        )

    print(f"{'='*70}\n")

    return dataloaders


if __name__ == "__main__":
    # 测试 DataLoader
    from config_utils import load_config

    config = load_config()
    config.print_config()

    # 创建 DataLoader
    dataloaders = create_multi_emotion_dataloaders(config)

    # 测试 seen emotions
    print("\n【测试 Seen Emotions DataLoader】")
    train_loader = dataloaders['train_seen']
    batch = next(iter(train_loader))

    print(f"批次数据形状:")
    print(f"  audio_features: {batch['audio_features'].shape}")
    print(f"  text_features: {batch['text_features'].shape}")
    print(f"  video_features: {batch['video_features'].shape}")
    print(f"  masks: {batch['masks'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  original_labels: {batch['original_labels'].shape}")
    print(f"  is_seen: {batch['is_seen']}")

    print(f"\n标签信息:")
    print(f"  映射后标签: {batch['labels'][:5]}")
    print(f"  原始标签: {batch['original_labels'][:5]}")

    # 测试 unseen emotions
    if 'train_unseen' in dataloaders:
        print("\n【测试 Unseen Emotions DataLoader】")
        unseen_loader = dataloaders['train_unseen']
        unseen_batch = next(iter(unseen_loader))

        print(f"批次数据形状:")
        print(f"  audio_features: {unseen_batch['audio_features'].shape}")
        print(f"  labels (应该是 -1): {unseen_batch['labels'][:5]}")
        print(f"  original_labels: {unseen_batch['original_labels'][:5]}")
        print(f"  is_seen: {unseen_batch['is_seen']}")
