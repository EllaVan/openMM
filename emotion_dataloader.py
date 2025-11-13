"""
情感数据集的PyTorch DataLoader
支持MOSEI和MELD数据集，根据emotion和label_id加载对应的数据
"""

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Union, Tuple


class EmotionDataset(Dataset):
    """
    情感数据集类，用于加载MOSEI或MELD的pkl文件

    数据格式:
    - MELD: MELD_{split}{emotion}label{label_id}.pkl
    - MOSEI: MOSEI{emotion}label{label_id}.pkl
    """

    def __init__(
        self,
        data_dir: str,
        dataset_name: str,
        emotion: str,
        label_id: int,
        split: Optional[str] = None,
        transform=None
    ):
        """
        Args:
            data_dir (str): 数据文件所在目录
            dataset_name (str): 数据集名称，'MELD' 或 'MOSEI'
            emotion (str): 情感类型，如 'happy', 'sad', 'anger' 等
            label_id (int): 标签ID
            split (str, optional): 数据集划分，如 'train', 'dev', 'test' (仅MELD需要)
            transform (callable, optional): 数据转换函数
        """
        self.data_dir = data_dir
        self.dataset_name = dataset_name.upper()
        self.emotion = emotion
        self.label_id = label_id
        self.split = split
        self.transform = transform

        # 验证数据集名称
        if self.dataset_name not in ['MELD', 'MOSEI']:
            raise ValueError(f"不支持的数据集: {dataset_name}. 请使用 'MELD' 或 'MOSEI'")

        # MELD数据集需要split参数
        if self.dataset_name == 'MELD' and split is None:
            raise ValueError("MELD数据集需要指定split参数 (train/dev/test)")

        # 加载数据
        self.data = self._load_data()

    def _load_data(self) -> List:
        """加载pkl文件"""
        # 构建文件名
        if self.dataset_name == 'MELD':
            filename = f"MELD_{self.split}{self.emotion}label{self.label_id}.pkl"
        else:  # MOSEI
            filename = f"MOSEI{self.emotion}label{self.label_id}.pkl"

        file_path = os.path.join(self.data_dir, filename)

        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")

        # 加载pkl文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        # 如果数据是字典格式，转换为列表
        if isinstance(data, dict):
            data = list(data.values()) if len(data) > 0 else []

        return data

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
            'dataset': self.dataset_name,
            'emotion': self.emotion,
            'label_id': self.label_id,
            'split': self.split,
            'size': len(self.data),
            'sample_keys': list(self.data[0].keys()) if len(self.data) > 0 and isinstance(self.data[0], dict) else None
        }


def create_emotion_dataloader(
    data_dir: str,
    dataset_name: str,
    emotion: str,
    label_id: int,
    split: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    **kwargs
) -> DataLoader:
    """
    创建情感数据的DataLoader

    Args:
        data_dir (str): 数据文件所在目录
        dataset_name (str): 数据集名称，'MELD' 或 'MOSEI'
        emotion (str): 情感类型
        label_id (int): 标签ID
        split (str, optional): 数据集划分 (仅MELD需要)
        batch_size (int): 批次大小，默认32
        shuffle (bool): 是否打乱数据，默认True
        num_workers (int): 数据加载的进程数，默认4
        transform (callable, optional): 数据转换函数
        **kwargs: 其他传递给DataLoader的参数

    Returns:
        DataLoader: PyTorch DataLoader对象
    """
    dataset = EmotionDataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        emotion=emotion,
        label_id=label_id,
        split=split,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        **kwargs
    )

    return dataloader


def create_multiple_dataloaders(
    data_dir: str,
    dataset_name: str,
    emotion_label_pairs: List[Tuple[str, int]],
    split: Optional[str] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    批量创建多个DataLoader

    Args:
        data_dir (str): 数据文件所在目录
        dataset_name (str): 数据集名称，'MELD' 或 'MOSEI'
        emotion_label_pairs (List[Tuple[str, int]]): (emotion, label_id)元组列表
        split (str, optional): 数据集划分 (仅MELD需要)
        batch_size (int): 批次大小
        shuffle (bool): 是否打乱数据
        num_workers (int): 数据加载的进程数
        transform (callable, optional): 数据转换函数
        **kwargs: 其他传递给DataLoader的参数

    Returns:
        Dict[str, DataLoader]: 字典，key为'emotion_labelid'，value为DataLoader

    Example:
        >>> pairs = [('happy', 0), ('sad', 1), ('anger', 2)]
        >>> dataloaders = create_multiple_dataloaders('Data', 'MOSEI', pairs)
        >>> # 返回 {'happy_0': dataloader1, 'sad_1': dataloader2, 'anger_2': dataloader3}
    """
    dataloaders = {}

    for emotion, label_id in emotion_label_pairs:
        key = f"{emotion}_{label_id}"
        dataloader = create_emotion_dataloader(
            data_dir=data_dir,
            dataset_name=dataset_name,
            emotion=emotion,
            label_id=label_id,
            split=split,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            transform=transform,
            **kwargs
        )
        dataloaders[key] = dataloader

    return dataloaders


def create_all_splits_dataloaders(
    data_dir: str,
    emotion: str,
    label_id: int,
    batch_size: int = 32,
    num_workers: int = 4,
    transform=None,
    **kwargs
) -> Dict[str, DataLoader]:
    """
    为MELD数据集创建所有split的DataLoader (train/dev/test)

    Args:
        data_dir (str): 数据文件所在目录
        emotion (str): 情感类型
        label_id (int): 标签ID
        batch_size (int): 批次大小
        num_workers (int): 数据加载的进程数
        transform (callable, optional): 数据转换函数
        **kwargs: 其他传递给DataLoader的参数

    Returns:
        Dict[str, DataLoader]: 包含'train', 'dev', 'test'三个DataLoader的字典
    """
    splits = ['train', 'dev', 'test']
    dataloaders = {}

    for split in splits:
        dataloader = create_emotion_dataloader(
            data_dir=data_dir,
            dataset_name='MELD',
            emotion=emotion,
            label_id=label_id,
            split=split,
            batch_size=batch_size,
            shuffle=(split == 'train'),  # 只在训练集打乱
            num_workers=num_workers,
            transform=transform,
            **kwargs
        )
        dataloaders[split] = dataloader

    return dataloaders


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
