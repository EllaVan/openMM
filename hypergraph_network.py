"""
完整的多模态超图融合网络

基于 hyper_graph_fusion_instruct.md 实现
包含单模态编码器、超图融合、对比学习和分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional

from hypergraph_modules import (
    MultimodalHypergraphLayer,
    GraphContrastiveLearning,
    BottleneckLayer
)


class UnimodalEncoder(nn.Module):
    """
    单模态编码器

    使用 Bi-LSTM 编码时序特征
    公式: S_m = FC(BiLSTM(U_m; θ^lstm_m))
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 1,
        dropout: float = 0.1,
        bidirectional: bool = True
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dim: LSTM 隐藏层维度
            output_dim: 输出特征维度
            num_layers: LSTM 层数
            dropout: Dropout 率
            bidirectional: 是否使用双向 LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        # Bi-LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # 全连接层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]

        Returns:
            out: [batch_size, seq_len, output_dim]
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # [batch, seq_len, hidden_dim*2]

        # 全连接
        out = self.fc(lstm_out)  # [batch, seq_len, output_dim]

        return out


class MultimodalHypergraphFusion(nn.Module):
    """
    完整的多模态超图融合网络

    组件:
    1. 单模态编码器 (Bi-LSTM)
    2. 多模态超图层
    3. 图对比学习
    4. Bottleneck 层
    5. 分类器
    """

    def __init__(
        self,
        text_input_dim: int,
        audio_input_dim: int,
        video_input_dim: int,
        encoder_hidden_dim: int = 128,
        encoder_output_dim: int = 256,
        hypergraph_hidden_dim: int = 256,
        num_hyperedges: int = 64,
        num_conv_layers: int = 2,
        bottleneck_dim: int = 128,
        num_classes: int = 7,
        dropout: float = 0.1,
        hyperedge_drop_rate: float = 0.2,
        use_contrastive: bool = True,
        contrastive_weight: float = 0.1,
        use_bottleneck: bool = True
    ):
        """
        Args:
            text_input_dim: 文本输入维度
            audio_input_dim: 音频输入维度
            video_input_dim: 视频输入维度
            encoder_hidden_dim: 编码器 LSTM 隐藏维度
            encoder_output_dim: 编码器输出维度
            hypergraph_hidden_dim: 超图隐藏维度
            num_hyperedges: 超边数量
            num_conv_layers: 超图卷积层数
            bottleneck_dim: Bottleneck 维度
            num_classes: 分类类别数
            dropout: Dropout 率
            hyperedge_drop_rate: 超边删除率
            use_contrastive: 是否使用对比学习
            contrastive_weight: 对比学习损失权重
            use_bottleneck: 是否使用 Bottleneck 层
        """
        super().__init__()

        self.use_contrastive = use_contrastive
        self.contrastive_weight = contrastive_weight
        self.use_bottleneck = use_bottleneck

        # 1. 单模态编码器
        self.text_encoder = UnimodalEncoder(
            input_dim=text_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        self.audio_encoder = UnimodalEncoder(
            input_dim=audio_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        self.video_encoder = UnimodalEncoder(
            input_dim=video_input_dim,
            hidden_dim=encoder_hidden_dim,
            output_dim=encoder_output_dim,
            dropout=dropout
        )

        # 2. 多模态超图层
        self.hypergraph = MultimodalHypergraphLayer(
            text_dim=encoder_output_dim,
            audio_dim=encoder_output_dim,
            video_dim=encoder_output_dim,
            hidden_dim=hypergraph_hidden_dim,
            num_hyperedges=num_hyperedges,
            num_conv_layers=num_conv_layers,
            dropout=dropout,
            hyperedge_drop_rate=hyperedge_drop_rate
        )

        # 3. Bottleneck 层（可选）
        if use_bottleneck:
            self.bottleneck = BottleneckLayer(
                in_dim=hypergraph_hidden_dim,
                bottleneck_dim=bottleneck_dim,
                out_dim=hypergraph_hidden_dim
            )

        # 4. 图对比学习（可选）
        if use_contrastive:
            self.contrastive = GraphContrastiveLearning(
                feature_dim=hypergraph_hidden_dim,
                projection_dim=128
            )

        # 5. 分类器
        # 将三个模态的节点特征聚合
        self.classifier = nn.Sequential(
            nn.Linear(hypergraph_hidden_dim * 3, hypergraph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hypergraph_hidden_dim, num_classes)
        )

        # 权重正则化
        self.l2_reg_weight = 0.001

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features: [batch_size, T, text_input_dim]
            audio_features: [batch_size, T, audio_input_dim]
            video_features: [batch_size, T, video_input_dim]
            labels: [batch_size] (可选，用于对比学习)

        Returns:
            dict: {
                'logits': 分类 logits [batch_size, num_classes],
                'loss': 总损失 (如果提供 labels),
                'cls_loss': 分类损失,
                'contrastive_loss': 对比学习损失,
                'H': 超图连接矩阵 (可选)
            }
        """
        batch_size, T, _ = text_features.shape

        # Step 1: 单模态编码
        text_encoded = self.text_encoder(text_features)  # [batch, T, encoder_out_dim]
        audio_encoded = self.audio_encoder(audio_features)
        video_encoded = self.video_encoder(video_features)

        # Step 2: 超图融合
        hypergraph_out = self.hypergraph(
            text_features=text_encoded,
            audio_features=audio_encoded,
            video_features=video_encoded,
            return_H=True
        )

        fused_features = hypergraph_out['fused']  # [batch, 3T, hidden_dim]
        H = hypergraph_out['H']  # [batch, 3T, num_hyperedges]

        # Step 3: Bottleneck（可选）
        if self.use_bottleneck:
            fused_features = self.bottleneck(fused_features)

        # Step 4: 分离三个模态的节点
        text_nodes = fused_features[:, :T, :]  # [batch, T, hidden_dim]
        audio_nodes = fused_features[:, T:2*T, :]
        video_nodes = fused_features[:, 2*T:3*T, :]

        # Step 5: 聚合节点特征（平均池化）
        text_pooled = text_nodes.mean(dim=1)  # [batch, hidden_dim]
        audio_pooled = audio_nodes.mean(dim=1)
        video_pooled = video_nodes.mean(dim=1)

        # 拼接三个模态
        multimodal_feature = torch.cat([
            text_pooled,
            audio_pooled,
            video_pooled
        ], dim=1)  # [batch, hidden_dim * 3]

        # Step 6: 分类
        logits = self.classifier(multimodal_feature)  # [batch, num_classes]

        # 准备输出
        output = {
            'logits': logits,
            'H': H,
            'multimodal_feature': multimodal_feature
        }

        # 如果提供标签，计算损失
        if labels is not None:
            # 分类损失
            cls_loss = F.cross_entropy(logits, labels)
            output['cls_loss'] = cls_loss

            total_loss = cls_loss

            # 对比学习损失
            if self.use_contrastive and self.training:
                # 使用聚合后的多模态特征进行对比学习
                contrastive_loss = self.contrastive(multimodal_feature, labels)
                output['contrastive_loss'] = contrastive_loss

                total_loss = total_loss + self.contrastive_weight * contrastive_loss

            # L2 正则化
            l2_reg = 0
            for param in self.parameters():
                l2_reg += torch.norm(param, p=2)

            total_loss = total_loss + self.l2_reg_weight * l2_reg
            output['reg_loss'] = self.l2_reg_weight * l2_reg

            output['loss'] = total_loss

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
            text_features: [batch_size, T, text_input_dim]
            audio_features: [batch_size, T, audio_input_dim]
            video_features: [batch_size, T, video_input_dim]

        Returns:
            predictions: [batch_size] 预测的类别
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(text_features, audio_features, video_features)
            logits = output['logits']
            predictions = torch.argmax(logits, dim=1)

        return predictions


class HypergraphEmotionClassifier(nn.Module):
    """
    基于超图的情感分类器

    简化的接口，自动处理不同输入格式
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        num_classes: int = 7,
        config: Optional[Dict] = None
    ):
        """
        Args:
            feature_dims: 特征维度字典 {'text': dim, 'audio': dim, 'video': dim}
            num_classes: 分类类别数
            config: 可选配置字典
        """
        super().__init__()

        # 默认配置
        default_config = {
            'encoder_hidden_dim': 128,
            'encoder_output_dim': 256,
            'hypergraph_hidden_dim': 256,
            'num_hyperedges': 64,
            'num_conv_layers': 2,
            'bottleneck_dim': 128,
            'dropout': 0.1,
            'hyperedge_drop_rate': 0.2,
            'use_contrastive': True,
            'contrastive_weight': 0.1,
            'use_bottleneck': True
        }

        if config:
            default_config.update(config)

        # 创建模型
        self.model = MultimodalHypergraphFusion(
            text_input_dim=feature_dims['text'],
            audio_input_dim=feature_dims['audio'],
            video_input_dim=feature_dims['video'],
            num_classes=num_classes,
            **default_config
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: {
                'audio_features': [batch, T, audio_dim],
                'text_features': [batch, T, text_dim],
                'video_features': [batch, T, video_dim],
                'label': [batch] (可选)
            }

        Returns:
            输出字典
        """
        return self.model(
            text_features=batch['text_features'],
            audio_features=batch['audio_features'],
            video_features=batch['video_features'],
            labels=batch.get('label')
        )

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """预测"""
        return self.model.predict(
            text_features=batch['text_features'],
            audio_features=batch['audio_features'],
            video_features=batch['video_features']
        )


# 导出
__all__ = [
    'UnimodalEncoder',
    'MultimodalHypergraphFusion',
    'HypergraphEmotionClassifier'
]
