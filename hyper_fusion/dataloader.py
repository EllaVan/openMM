"""
超图融合数据加载器 - 支持 Padding + Masking
用于处理变长序列，自动填充到批次内最大长度
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import random


class PaddedEmotionDataset(Dataset):
    """
    情感数据集 - 支持变长序列

    数据格式:
    - MELD: MELD_{split}{emotion}label{label_id}.pkl
    - MOSEI: MOSEI{emotion}label{label_id}.pkl

    每个样本包含:
    - audio_features: [num_frames, audio_dim]
    - text_features: [num_frames, text_dim]
    - video_features: [num_frames, video_dim]
    - label: int
    - num_frames: int
    """

    def __init__(self, data: List[Dict]):
        """
        Args:
            data: 数据列表，每个元素为字典
        """
        self.data = data

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
                'label': int,
                'num_frames': int,
                'sample_id': str (可选)
            }
        """
        return self.data[idx]

    def get_stats(self) -> Dict:
        """获取数据集统计信息"""
        num_frames_list = [item['num_frames'] for item in self.data]

        import numpy as np
        return {
            'size': len(self.data),
            'avg_frames': np.mean(num_frames_list),
            'median_frames': np.median(num_frames_list),
            'min_frames': np.min(num_frames_list),
            'max_frames': np.max(num_frames_list),
            'std_frames': np.std(num_frames_list)
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
            'labels': [batch_size],
            'num_frames': [batch_size]  # 每个样本的实际帧数
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
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    num_frames_list = torch.tensor([item['num_frames'] for item in batch], dtype=torch.long)

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
        'num_frames': num_frames_list
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


def load_mosei_data(
    data_dir: str,
    emotion: str,
    label_id: int,
    train_ratio: float = 0.7,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    加载 MOSEI 数据集并划分

    Args:
        data_dir: 数据目录
        emotion: 情感类型
        label_id: 标签 ID
        train_ratio: 训练集比例
        seed: 随机种子

    Returns:
        train_data, test_data
    """
    filename = f"MOSEI{emotion}label{label_id}.pkl"
    file_path = os.path.join(data_dir, filename)

    all_data = load_pkl_file(file_path)

    # 设置随机种子
    random.seed(seed)
    indices = list(range(len(all_data)))
    random.shuffle(indices)

    # 划分数据
    train_size = int(len(all_data) * train_ratio)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = [all_data[i] for i in train_indices]
    test_data = [all_data[i] for i in test_indices]

    return train_data, test_data


def load_meld_data(
    data_dir: str,
    emotion: str,
    label_id: int
) -> Tuple[List[Dict], List[Dict]]:
    """
    加载 MELD 数据集
    - 训练集: train + dev 合并
    - 测试集: test

    Args:
        data_dir: 数据目录
        emotion: 情感类型
        label_id: 标签 ID

    Returns:
        train_data, test_data
    """
    train_file = f"MELD_train{emotion}label{label_id}.pkl"
    dev_file = f"MELD_dev{emotion}label{label_id}.pkl"
    test_file = f"MELD_test{emotion}label{label_id}.pkl"

    train_data = load_pkl_file(os.path.join(data_dir, train_file))
    dev_data = load_pkl_file(os.path.join(data_dir, dev_file))
    test_data = load_pkl_file(os.path.join(data_dir, test_file))

    # 合并 train 和 dev
    train_data_merged = train_data + dev_data

    return train_data_merged, test_data


def create_dataloaders(
    data_dir: str,
    dataset_name: str,
    emotion: str,
    label_id: int,
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    seed: int = 42,
    shuffle_train: bool = True,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    创建支持 Padding + Masking 的 DataLoader

    Args:
        data_dir: 数据目录
        dataset_name: 'MOSEI' 或 'MELD'
        emotion: 情感类型
        label_id: 标签 ID
        batch_size: 批次大小
        num_workers: 数据加载进程数
        train_ratio: 训练集比例（MOSEI）
        seed: 随机种子（MOSEI）
        shuffle_train: 是否打乱训练集
        pin_memory: 是否使用 pin_memory

    Returns:
        dataloaders: {
            'train': DataLoader,
            'test': DataLoader
        }
    """
    dataset_name = dataset_name.upper()

    if dataset_name not in ['MOSEI', 'MELD']:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 加载数据
    if dataset_name == 'MOSEI':
        train_data, test_data = load_mosei_data(
            data_dir, emotion, label_id, train_ratio, seed
        )
    else:  # MELD
        train_data, test_data = load_meld_data(
            data_dir, emotion, label_id
        )

    # 创建 Dataset
    train_dataset = PaddedEmotionDataset(train_data)
    test_dataset = PaddedEmotionDataset(test_data)

    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name} - {emotion} (label_id={label_id})")
    print(f"{'='*60}")
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    train_stats = train_dataset.get_stats()
    print(f"\n训练集帧数统计:")
    print(f"  平均: {train_stats['avg_frames']:.1f}")
    print(f"  中位数: {train_stats['median_frames']:.1f}")
    print(f"  范围: [{train_stats['min_frames']:.0f}, {train_stats['max_frames']:.0f}]")
    print(f"  标准差: {train_stats['std_frames']:.1f}")
    print(f"{'='*60}\n")

    # 创建 DataLoader（使用 padded_collate_fn）
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=padded_collate_fn,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=padded_collate_fn,
        pin_memory=pin_memory
    )

    return {
        'train': train_loader,
        'test': test_loader
    }


if __name__ == "__main__":
    # 测试 DataLoader
    dataloaders = create_dataloaders(
        data_dir='./output/mosei_features',
        dataset_name='MOSEI',
        emotion='happy',
        label_id=0,
        batch_size=4
    )

    # 获取一个批次
    train_loader = dataloaders['train']
    batch = next(iter(train_loader))

    print("批次数据形状:")
    print(f"  audio_features: {batch['audio_features'].shape}")
    print(f"  text_features: {batch['text_features'].shape}")
    print(f"  video_features: {batch['video_features'].shape}")
    print(f"  masks: {batch['masks'].shape}")
    print(f"  labels: {batch['labels'].shape}")
    print(f"  num_frames: {batch['num_frames']}")

    print(f"\n每个样本的有效帧数:")
    for i, nf in enumerate(batch['num_frames']):
        print(f"  样本 {i}: {nf} 帧")
