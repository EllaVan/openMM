"""
配置文件读取和管理工具
"""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


class Config:
    """配置管理类"""

    def __init__(self, config_path: str = None):
        """
        Args:
            config_path: 配置文件路径，如果为 None 则使用默认路径
        """
        if config_path is None:
            # 默认配置文件路径
            config_path = os.path.join(
                os.path.dirname(__file__),
                'config.yaml'
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")

        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self._config = yaml.safe_load(f)

        self.config_path = config_path

    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置项

        Args:
            key: 配置键，支持点号分隔的嵌套键 (e.g., 'dataset.name')
            default: 默认值

        Returns:
            配置值
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def __getitem__(self, key: str) -> Any:
        """支持字典式访问"""
        return self.get(key)

    @property
    def dataset(self) -> Dict:
        """获取数据集配置"""
        return self._config.get('dataset', {})

    @property
    def dataloader(self) -> Dict:
        """获取数据加载器配置"""
        return self._config.get('dataloader', {})

    @property
    def model(self) -> Dict:
        """获取模型配置"""
        return self._config.get('model', {})

    @property
    def training(self) -> Dict:
        """获取训练配置"""
        return self._config.get('training', {})

    @property
    def system(self) -> Dict:
        """获取系统配置"""
        return self._config.get('system', {})

    @property
    def experiment(self) -> Dict:
        """获取实验配置"""
        return self._config.get('experiment', {})

    def get_seen_emotions(self) -> Dict[str, int]:
        """
        获取 seen emotions 配置

        Returns:
            dict: {emotion_name: label_id}
        """
        return self.dataset.get('seen_emotions', {})

    def get_unseen_emotions(self) -> Dict[str, int]:
        """
        获取 unseen emotions 配置

        Returns:
            dict: {emotion_name: label_id}
        """
        return self.dataset.get('unseen_emotions', {})

    def get_all_emotions(self) -> Dict[str, int]:
        """
        获取所有 emotions (seen + unseen)

        Returns:
            dict: {emotion_name: label_id}
        """
        emotions = {}
        emotions.update(self.get_seen_emotions())
        emotions.update(self.get_unseen_emotions())
        return emotions

    def get_num_classes(self) -> int:
        """
        获取分类类别数 (基于 seen emotions)

        Returns:
            int: 类别数
        """
        seen_emotions = self.get_seen_emotions()
        return len(seen_emotions)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return self._config.copy()

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"

    def print_config(self):
        """打印配置信息"""
        print(f"\n{'='*70}")
        print(f"配置文件: {self.config_path}")
        print(f"{'='*70}")

        # 数据集配置
        print("\n【数据集配置】")
        print(f"  数据集名称: {self.dataset.get('name')}")
        print(f"  数据目录: {self.dataset.get('data_dir')}")
        print(f"  Seen emotions: {self.get_seen_emotions()}")
        print(f"  Unseen emotions: {self.get_unseen_emotions()}")
        print(f"  训练集比例: {self.dataset.get('train_ratio')}")

        # 数据加载配置
        print("\n【数据加载配置】")
        print(f"  Batch size: {self.dataloader.get('batch_size')}")
        print(f"  Num workers: {self.dataloader.get('num_workers')}")
        print(f"  Shuffle train: {self.dataloader.get('shuffle_train')}")

        # 模型配置
        print("\n【模型配置】")
        print(f"  编码器隐藏层维度: {self.model.get('encoder', {}).get('hidden_dim')}")
        print(f"  编码器输出维度: {self.model.get('encoder', {}).get('output_dim')}")
        print(f"  超边数量: {self.model.get('hypergraph', {}).get('num_hyperedges')}")
        print(f"  超图卷积层数: {self.model.get('hypergraph', {}).get('num_conv_layers')}")
        print(f"  使用 Bottleneck: {self.model.get('bottleneck', {}).get('use_bottleneck')}")
        print(f"  使用对比学习: {self.model.get('contrastive', {}).get('use_contrastive')}")

        # 训练配置
        print("\n【训练配置】")
        print(f"  训练轮数: {self.training.get('epochs')}")
        print(f"  学习率: {self.training.get('learning_rate')}")
        print(f"  权重衰减: {self.training.get('weight_decay')}")

        # 系统配置
        print("\n【系统配置】")
        print(f"  设备: {self.system.get('device')}")
        print(f"  随机种子: {self.system.get('random_seed')}")
        print(f"  保存目录: {self.system.get('save_dir')}")

        # 实验配置
        print("\n【实验配置】")
        print(f"  实验名称: {self.experiment.get('name')}")
        print(f"  实验描述: {self.experiment.get('description')}")

        print(f"{'='*70}\n")


def load_config(config_path: str = None) -> Config:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        Config 对象
    """
    return Config(config_path)


if __name__ == "__main__":
    # 测试配置加载
    config = load_config()
    config.print_config()

    # 测试获取配置
    print("\n测试配置访问:")
    print(f"数据集名称: {config.get('dataset.name')}")
    print(f"Batch size: {config.get('dataloader.batch_size')}")
    print(f"Seen emotions: {config.get_seen_emotions()}")
    print(f"分类类别数: {config.get_num_classes()}")
