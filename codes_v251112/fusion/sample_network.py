"""
基于样本级别超图的多模态情感分类网络

网络结构：
1. 时序编码器 (Bi-LSTM) - 编码每个样本的时序特征
2. 时序池化 - 将时序特征池化为样本级别特征
3. 样本级别超图融合 - 在batch内构建样本超图
4. 分类器 - 基于样本表示进行分类
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


class TemporalEncoder(nn.Module):
    """
    时序编码器 - 使用 Bi-LSTM 编码时序特征
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM隐藏层维度
            output_dim: 输出特征维度
            num_layers: LSTM层数
            dropout: Dropout率
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        # 全连接层
        lstm_output_dim = hidden_dim * 2  # 双向
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] - True表示有效位置

        Returns:
            out: [batch_size, seq_len, output_dim]
        """
        # 如果有mask，使用pack_padded_sequence
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()  # [batch_size]

            # Pack序列
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )

            # LSTM
            packed_out, _ = self.lstm(packed)

            # Unpack
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True
            )
        else:
            lstm_out, _ = self.lstm(x)

        # 全连接
        out = self.fc(lstm_out)

        # 确保填充位置为0
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()

        return out


class TemporalPooling(nn.Module):
    """
    时序池化模块 - 将时序特征池化为样本级别特征
    """

    def __init__(self, pooling_type: str = 'masked_mean'):
        """
        Args:
            pooling_type: 池化类型
                - 'masked_mean': 带mask的平均池化
                - 'max': 最大池化
                - 'last': 使用最后一个有效帧
        """
        super().__init__()
        self.pooling_type = pooling_type

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, dim]
            mask: [batch_size, seq_len] - True表示有效位置

        Returns:
            pooled: [batch_size, dim]
        """
        if self.pooling_type == 'masked_mean':
            if mask is not None:
                # 扩展mask
                mask_expanded = mask.unsqueeze(-1).float()  # [batch, seq_len, 1]

                # 计算有效帧数
                valid_counts = mask.sum(dim=1, keepdim=True).float()  # [batch, 1]

                # Masked平均
                pooled = (x * mask_expanded).sum(dim=1) / (valid_counts + 1e-10)
            else:
                # 普通平均
                pooled = x.mean(dim=1)

        elif self.pooling_type == 'max':
            if mask is not None:
                # 将填充位置设为极小值
                x_masked = x.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                pooled = x_masked.max(dim=1)[0]
            else:
                pooled = x.max(dim=1)[0]

        elif self.pooling_type == 'last':
            if mask is not None:
                # 获取每个样本的最后一个有效帧索引
                lengths = mask.sum(dim=1) - 1  # [batch]
                batch_size = x.shape[0]
                pooled = x[torch.arange(batch_size), lengths]
            else:
                pooled = x[:, -1, :]

        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")

        return pooled


class SampleHypergraphNetwork(nn.Module):
    """
    基于样本级别超图的多模态情感分类网络
    """

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        video_input_dim: int,
        encoder_hidden_dim: int = 256,
        encoder_output_dim: int = 256,
        hypergraph_hidden_dim: int = 256,
        num_conv_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.1,
        pooling_type: str = 'masked_mean',
        use_edge_weights: bool = True,
        similarity_temperature: float = 1.0
    ):
        """
        Args:
            text_input_dim: 文本输入维度
            audio_input_dim: 音频输入维度
            video_input_dim: 视频输入维度
            encoder_hidden_dim: 编码器隐藏层维度
            encoder_output_dim: 编码器输出维度
            hypergraph_hidden_dim: 超图隐藏层维度
            num_conv_layers: 超图卷积层数
            num_classes: 分类类别数
            dropout: Dropout率
            pooling_type: 池化类型
            use_edge_weights: 是否使用边权重
            similarity_temperature: 相似度温度参数
        """
        super().__init__()

        self.num_classes = num_classes

        # 1. 时序编码器
        self.text_encoder = TemporalEncoder(
            input_dim=text_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        self.audio_encoder = TemporalEncoder(
            input_dim=audio_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        self.video_encoder = TemporalEncoder(
            input_dim=video_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        # 2. 时序池化
        self.text_pooling = TemporalPooling(pooling_type=pooling_type)
        self.audio_pooling = TemporalPooling(pooling_type=pooling_type)
        self.video_pooling = TemporalPooling(pooling_type=pooling_type)

        # 3. 样本级别超图
        self.hypergraph = SampleLevelHypergraph(
            text_dim=encoder_output_dim,
            audio_dim=encoder_output_dim,
            video_dim=encoder_output_dim,
            hidden_dim=hypergraph_hidden_dim,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            use_edge_weights=use_edge_weights,
            similarity_temperature=similarity_temperature
        )

        # 4. 分类器
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
        masks: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            text_features: [batch_size, T, text_input_dim]
            audio_features: [batch_size, T, audio_input_dim]
            video_features: [batch_size, T, video_input_dim]
            masks: [batch_size, T] - 有效帧mask
            labels: [batch_size] - 标签（可选）

        Returns:
            dict: {
                'logits': 分类logits [batch_size, num_classes],
                'loss': 总损失 (如果提供labels),
                'sample_features': 样本表示 [batch_size, hidden_dim]
            }
        """
        # 1. 时序编码
        text_encoded = self.text_encoder(text_features, mask=masks)
        audio_encoded = self.audio_encoder(audio_features, mask=masks)
        video_encoded = self.video_encoder(video_features, mask=masks)

        # 2. 时序池化
        text_pooled = self.text_pooling(text_encoded, mask=masks)
        audio_pooled = self.audio_pooling(audio_encoded, mask=masks)
        video_pooled = self.video_pooling(video_encoded, mask=masks)

        # 3. 样本级别超图融合
        hypergraph_out = self.hypergraph(
            text_pooled,
            audio_pooled,
            video_pooled,
            return_H=False
        )

        sample_features = hypergraph_out['sample_features']

        # 4. 分类
        logits = self.classifier(sample_features)

        # 准备输出
        output = {
            'logits': logits,
            'sample_features': sample_features
        }

        # 5. 计算损失（如果提供标签）
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            output['loss'] = loss

        return output

    def predict(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        masks: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测

        Args:
            text_features: [batch_size, T, text_input_dim]
            audio_features: [batch_size, T, audio_input_dim]
            video_features: [batch_size, T, video_input_dim]
            masks: [batch_size, T]

        Returns:
            predictions: [batch_size] - 预测类别
        """
        output = self.forward(text_features, audio_features, video_features, masks)
        predictions = torch.argmax(output['logits'], dim=1)
        return predictions


class SampleHypergraphClassifier(nn.Module):
    """
    样本级别超图情感分类器 - 简化接口

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
            'encoder_hidden_dim': 256,
            'encoder_output_dim': 256,
            'hypergraph_hidden_dim': 256,
            'num_conv_layers': 2,
            'dropout': 0.1,
            'pooling_type': 'masked_mean',
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
                    'text_features': [batch, T, text_dim],
                    'audio_features': [batch, T, audio_dim],
                    'video_features': [batch, T, video_dim],
                    'masks': [batch, T],
                    'labels': [batch]
                }

        Returns:
            output: 模型输出
        """
        return self.model(
            text_features=batch['text_features'],
            audio_features=batch['audio_features'],
            video_features=batch['video_features'],
            masks=batch['masks'],
            labels=batch.get('labels', None)
        )

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测"""
        return self.model.predict(
            text_features=batch['text_features'],
            audio_features=batch['audio_features'],
            video_features=batch['video_features'],
            masks=batch['masks']
        )


if __name__ == "__main__":
    # 测试网络
    batch_size = 4
    T = 50
    feature_dims = {'text': 768, 'audio': 768, 'video': 768}

    # 创建测试批次
    batch = {
        'text_features': torch.randn(batch_size, T, 768),
        'audio_features': torch.randn(batch_size, T, 768),
        'video_features': torch.randn(batch_size, T, 768),
        'masks': torch.ones(batch_size, T, dtype=torch.bool),
        'labels': torch.randint(0, 2, (batch_size,))
    }

    # 模拟变长序列
    batch['masks'][0, 30:] = False
    batch['masks'][1, 40:] = False
    batch['masks'][3, 25:] = False

    # 创建模型
    model = SampleHypergraphClassifier(
        feature_dims=feature_dims,
        num_classes=2
    )

    # 前向传播
    output = model(batch)

    print("=" * 70)
    print("样本级别超图网络测试")
    print("=" * 70)
    print(f"\n输出:")
    print(f"  logits: {output['logits'].shape}")
    print(f"  loss: {output['loss'].item():.4f}")
    print(f"  sample_features: {output['sample_features'].shape}")

    # 预测
    predictions = model.predict(batch)
    print(f"\n预测: {predictions}")
    print(f"真实: {batch['labels']}")

    print("\n" + "=" * 70)
