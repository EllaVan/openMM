"""
情感数据集的PyTorch DataLoader
支持MOSEI和MELD数据集，根据emotion和label_id加载对应的数据

数据加载策略：
- MOSEI: 加载指定emotion和label_id的数据，按7/3划分训练集和测试集
- MELD: train+dev合并为训练集，test为测试集
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Optional, Union, Tuple
import random


class EmotionDataset(Dataset):
    """
    情感数据集类，用于加载MOSEI或MELD的pkl文件

    数据格式:
    - MELD: MELD_{split}{emotion}label{label_id}.pkl
    - MOSEI: MOSEI{emotion}label{label_id}.pkl
    """

    def __init__(
        self,
        data: List,
        transform=None
    ):
        """
        Args:
            data (List): 数据列表
            transform (callable, optional): 数据转换函数
        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[Dict, Tuple]:
        """
        获取单个样本

        Args:
            idx (int): 样本索引

        Returns:
            样本数据，可能包含audio_features, text_features, video_features, label等
        """
        sample = self.data[idx]

        # 应用transform
        if self.transform:
            sample = self.transform(sample)

        return sample

    def get_info(self) -> Dict:
        """获取数据集信息"""
        return {
            'size': len(self.data),
            'sample_keys': list(self.data[0].keys()) if len(self.data) > 0 and isinstance(self.data[0], dict) else None
        }


def load_pkl_file(file_path: str) -> List:
    """
    加载pkl文件并返回数据列表

    Args:
        file_path (str): pkl文件路径

    Returns:
        List: 数据列表
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # 如果数据是字典格式，转换为列表
    if isinstance(data, dict):
        data = list(data.values()) if len(data) > 0 else []

    return data


def load_mosei_data(
    data_dir: str,
    emotion: str,
    label_id: int,
    train_ratio: float = 0.7,
    seed: int = 42
) -> Tuple[List, List]:
    """
    加载MOSEI数据集，并按照指定比例分割为训练集和测试集

    Args:
        data_dir (str): 数据文件所在目录
        emotion (str): 情感类型
        label_id (int): 标签ID
        train_ratio (float): 训练集比例，默认0.7
        seed (int): 随机种子，默认42

    Returns:
        Tuple[List, List]: (训练数据, 测试数据)
    """
    filename = f"MOSEI{emotion}label{label_id}.pkl"
    file_path = os.path.join(data_dir, filename)

    # 加载所有数据
    all_data = load_pkl_file(file_path)

    # 设置随机种子
    random.seed(seed)

    # 打乱数据
    indices = list(range(len(all_data)))
    random.shuffle(indices)

    # 计算分割点
    train_size = int(len(all_data) * train_ratio)

    # 分割数据
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    train_data = [all_data[i] for i in train_indices]
    test_data = [all_data[i] for i in test_indices]

    return train_data, test_data


def load_meld_data(
    data_dir: str,
    emotion: str,
    label_id: int
) -> Tuple[List, List]:
    """
    加载MELD数据集
    - 训练集：train + dev 合并
    - 测试集：test

    Args:
        data_dir (str): 数据文件所在目录
        emotion (str): 情感类型
        label_id (int): 标签ID

    Returns:
        Tuple[List, List]: (训练数据, 测试数据)
    """
    # 加载train, dev, test数据
    train_file = f"MELD_train{emotion}label{label_id}.pkl"
    dev_file = f"MELD_dev{emotion}label{label_id}.pkl"
    test_file = f"MELD_test{emotion}label{label_id}.pkl"

    train_data = load_pkl_file(os.path.join(data_dir, train_file))
    dev_data = load_pkl_file(os.path.join(data_dir, dev_file))
    test_data = load_pkl_file(os.path.join(data_dir, test_file))

    # 合并train和dev
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
    transform=None,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    创建情感数据的DataLoader

    - MOSEI: 返回 {'train': train_loader, 'test': test_loader}
    - MELD: 返回 {'train': train_loader, 'test': test_loader}

    Args:
        data_dir (str): 数据文件所在目录
        dataset_name (str): 数据集名称，'MELD' 或 'MOSEI'
        emotion (str): 情感类型
        label_id (int): 标签ID
        batch_size (int): 批次大小，默认32
        num_workers (int): 数据加载的进程数，默认4
        train_ratio (float): 训练集比例（仅MOSEI），默认0.7
        seed (int): 随机种子（仅MOSEI），默认42
        transform (callable, optional): 数据转换函数
        **kwargs: 其他传递给DataLoader的参数

    Returns:
        Dict[str, DataLoader]: 包含'train'和'test'的DataLoader字典
    """
    dataset_name = dataset_name.upper()

    if dataset_name not in ['MELD', 'MOSEI']:
        raise ValueError(f"不支持的数据集: {dataset_name}. 请使用 'MELD' 或 'MOSEI'")

    # 加载数据
    if dataset_name == 'MOSEI':
        train_data, test_data = load_mosei_data(
            data_dir=data_dir,
            emotion=emotion,
            label_id=label_id,
            train_ratio=train_ratio,
            seed=seed
        )
    else:  # MELD
        train_data, test_data = load_meld_data(
            data_dir=data_dir,
            emotion=emotion,
            label_id=label_id
        )

    # 创建Dataset
    train_dataset = EmotionDataset(data=train_data, transform=transform)
    test_dataset = EmotionDataset(data=test_data, transform=transform)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        **kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        **kwargs
    )

    return {
        'train': train_loader,
        'test': test_loader
    }


def create_multiple_dataloaders(
    data_dir: str,
    dataset_name: str,
    emotion_label_pairs: List[Tuple[str, int]],
    batch_size: int = 32,
    num_workers: int = 4,
    train_ratio: float = 0.7,
    seed: int = 42,
    transform=None,
    **kwargs
) -> Dict[str, Dict[str, DataLoader]]:
    """
    批量创建多个DataLoader

    Args:
        data_dir (str): 数据文件所在目录
        dataset_name (str): 数据集名称，'MELD' 或 'MOSEI'
        emotion_label_pairs (List[Tuple[str, int]]): (emotion, label_id)元组列表
        batch_size (int): 批次大小
        num_workers (int): 数据加载的进程数
        train_ratio (float): 训练集比例（仅MOSEI）
        seed (int): 随机种子（仅MOSEI）
        transform (callable, optional): 数据转换函数
        **kwargs: 其他传递给DataLoader的参数

    Returns:
        Dict[str, Dict[str, DataLoader]]: 嵌套字典
            外层key为'emotion_labelid'，内层为{'train': loader, 'test': loader}

    Example:
        >>> pairs = [('happy', 0), ('sad', 1)]
        >>> loaders = create_multiple_dataloaders('Data', 'MOSEI', pairs)
        >>> train_loader = loaders['happy_0']['train']
        >>> test_loader = loaders['happy_0']['test']
    """
    all_dataloaders = {}

    for emotion, label_id in emotion_label_pairs:
        key = f"{emotion}_{label_id}"
        dataloaders = create_dataloaders(
            data_dir=data_dir,
            dataset_name=dataset_name,
            emotion=emotion,
            label_id=label_id,
            batch_size=batch_size,
            num_workers=num_workers,
            train_ratio=train_ratio,
            seed=seed,
            transform=transform,
            **kwargs
        )
        all_dataloaders[key] = dataloaders

    return all_dataloaders


# 自定义collate函数示例
def custom_collate_fn(batch):
    """
    自定义batch整理函数

    假设每个样本是一个字典，包含:
    - audio_features: tensor
    - text_features: tensor
    - video_features: tensor
    - label: int
    """
    audio_features = torch.stack([item['audio_features'] for item in batch])
    text_features = torch.stack([item['text_features'] for item in batch])
    video_features = torch.stack([item['video_features'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])

    return {
        'audio': audio_features,
        'text': text_features,
        'video': video_features,
        'labels': labels
    }
