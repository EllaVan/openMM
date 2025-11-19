"""
样本级别超图网络 - 直接使用样本特征（无时序）

特征输入: [batch_size, feature_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import sys
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from fusion.sample_hypergraph import SampleLevelHypergraph


class SampleHypergraphNetwork(nn.Module):
    """
    样本级别超图网络

    输入: 每个样本的每个模态特征 [batch_size, feature_dim]
    输出: 分类结果
    """

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        video_input_dim: int,
        hypergraph_hidden_dim: int = 256,
        num_conv_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        use_edge_weights: bool = True,
        similarity_temperature: float = 1.0
    ):
        """
        Args:
            text_input_dim: 文本输入维度
            audio_input_dim: 音频输入维度
            video_input_dim: 视频输入维度
            hypergraph_hidden_dim: 超图隐藏层维度
            num_conv_layers: 超图卷积层数
            num_classes: 分类类别数
            dropout: Dropout率
            use_edge_weights: 是否使用边权重
            similarity_temperature: 相似度温度参数
        """
        super().__init__()

        self.num_classes = num_classes

        # 样本级别超图
        self.hypergraph = SampleLevelHypergraph(
            text_dim=text_input_dim,
            audio_dim=audio_input_dim,
            video_dim=video_input_dim,
            hidden_dim=hypergraph_hidden_dim,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            use_edge_weights=use_edge_weights,
            similarity_temperature=similarity_temperature
        )

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hypergraph_hidden_dim, hypergraph_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hypergraph_hidden_dim // 2, num_classes)
        )

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            text_features: [batch_size, text_input_dim]
            audio_features: [batch_size, audio_input_dim]
            video_features: [batch_size, video_input_dim]
            labels: [batch_size] - 标签（可选）

        Returns:
            dict: {
                'logits': 分类logits [batch_size, num_classes],
                'loss': 总损失 (如果提供labels),
                'sample_features': 样本表示 [batch_size, hidden_dim]
            }
        """
        # 样本级别超图融合
        hypergraph_out = self.hypergraph(
            text_features,
            audio_features,
            video_features,
            return_H=False
        )

        sample_features = hypergraph_out['sample_features']

        # 分类
        logits = self.classifier(sample_features)

        # 准备输出
        output = {
            'logits': logits,
            'sample_features': sample_features
        }

        # 计算损失（如果提供标签）
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output['loss'] = loss

        return output

    def predict(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        """
        预测

        Args:
            text_features: [batch_size, text_input_dim]
            audio_features: [batch_size, audio_input_dim]
            video_features: [batch_size, video_input_dim]

        Returns:
            predictions: [batch_size] - 预测类别
        """
        output = self.forward(text_features, audio_features, video_features)
        predictions = torch.argmax(output['logits'], dim=1)
        return predictions


class SampleHypergraphClassifier(nn.Module):
    """
    样本级别超图分类器 - 简化接口

    方便与DataLoader集成
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        num_classes: int = 2,
        config: Optional[Dict] = None
    ):
        """
        Args:
            feature_dims: {'text': dim, 'audio': dim, 'video': dim}
            num_classes: 分类类别数
            config: 配置字典
        """
        super().__init__()

        # 默认配置
        default_config = {
            'hypergraph_hidden_dim': 768,
            'num_conv_layers': 2,
            'dropout': 0.1,
            'use_edge_weights': True,
            'similarity_temperature': 1.0
        }

        if config is not None:
            default_config.update(config)

        self.config = default_config

        # 创建模型
        self.model = SampleHypergraphNetwork(
            text_input_dim=feature_dims['text'],
            audio_input_dim=feature_dims['audio'],
            video_input_dim=feature_dims['video'],
            num_classes=num_classes,
            **default_config
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: 来自DataLoader的批次数据
                {
                    'text_features': [batch, text_dim],
                    'audio_features': [batch, audio_dim],
                    'video_features': [batch, video_dim],
                    'labels': [batch]
                }

        Returns:
            output: 模型输出
        """
        return self.model(
            text_features=batch['text_features'],
            audio_features=batch['audio_features'],
            video_features=batch['video_features'],
            labels=batch.get('labels', None)
        )

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测"""
        return self.model.predict(
            text_features=batch['text_features'],
            audio_features=batch['audio_features'],
            video_features=batch['video_features']
        )
