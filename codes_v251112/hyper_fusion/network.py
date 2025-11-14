"""
完整的多模态超图融合网络 - 支持 Padding + Masking

基于 hyper_graph_fusion_instruct.md 实现
包含单模态编码器、超图融合、对比学习和分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from hyper_fusion.modules import (
    MultimodalHypergraphLayer,
    GraphContrastiveLearning,
    BottleneckLayer
)


class UnimodalEncoder(nn.Module):
    """
    单模态编码器 - 支持 Mask

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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] - True 表示有效位置

        Returns:
            out: [batch_size, seq_len, output_dim]
        """
        # 如果有 mask，使用 pack_padded_sequence 提高效率
        if mask is not None:
            # 计算每个序列的有效长度
            lengths = mask.sum(dim=1).cpu()  # [batch_size]

            # Pack 序列
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
            # 没有 mask，直接处理
            lstm_out, _ = self.lstm(x)

        # 全连接
        out = self.fc(lstm_out)

        # 确保填充位置为 0
        if mask is not None:
            out = out * mask.unsqueeze(-1).float()

        return out


class MultimodalHypergraphFusion(nn.Module):
    """
    完整的多模态超图融合网络 - 支持 Mask

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
        encoder_hidden_dim: int = 256,
        encoder_output_dim: int = 256,
        hypergraph_hidden_dim: int = 256,
        num_hyperedges: int = 64,
        num_conv_layers: int = 2,
        bottleneck_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
        hyperedge_drop_rate: float = 0.2,
        use_contrastive: bool = True,
        use_bottleneck: bool = True,
        contrastive_weight: float = 0.1
    ):
        """
        Args:
            text_input_dim: 文本输入维度
            audio_input_dim: 音频输入维度
            video_input_dim: 视频输入维度
            encoder_hidden_dim: 编码器隐藏层维度
            encoder_output_dim: 编码器输出维度
            hypergraph_hidden_dim: 超图隐藏层维度
            num_hyperedges: 超边数量
            num_conv_layers: 超图卷积层数
            bottleneck_dim: Bottleneck 维度
            num_classes: 分类类别数
            dropout: Dropout 率
            hyperedge_drop_rate: 超边删除率
            use_contrastive: 是否使用对比学习
            use_bottleneck: 是否使用 Bottleneck 层
            contrastive_weight: 对比学习损失权重
        """
        super().__init__()

        self.num_classes = num_classes
        self.use_contrastive = use_contrastive
        self.use_bottleneck = use_bottleneck
        self.contrastive_weight = contrastive_weight

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
        self.classifier = nn.Sequential(
            nn.Linear(hypergraph_hidden_dim * 3, hypergraph_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hypergraph_hidden_dim, num_classes)
        )

        # L2 正则化权重
        self.l2_reg_weight = 0.001

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
            masks: [batch_size, T] - 有效帧 mask
            labels: [batch_size] - 标签（可选，用于对比学习和损失计算）

        Returns:
            dict: {
                'logits': 分类 logits [batch_size, num_classes],
                'loss': 总损失 (如果提供 labels),
                'cls_loss': 分类损失,
                'contrastive_loss': 对比学习损失 (如果启用),
                'l2_reg': L2 正则化项,
                'H': 超图连接矩阵 (可选)
            }
        """
        # 1. 单模态编码
        text_encoded = self.text_encoder(text_features, mask=masks)
        audio_encoded = self.audio_encoder(audio_features, mask=masks)
        video_encoded = self.video_encoder(video_features, mask=masks)

        # 2. 超图融合
        hypergraph_out = self.hypergraph(
            text_encoded,
            audio_encoded,
            video_encoded,
            mask=masks,
            return_H=True
        )

        fused_features = hypergraph_out['fused']  # [batch, 3T, hidden_dim]
        H = hypergraph_out['H']

        # 3. Bottleneck（可选）
        if self.use_bottleneck:
            fused_features, bottleneck_features = self.bottleneck(fused_features)

        # 4. 分离三个模态并池化
        batch_size, total_nodes, hidden_dim = fused_features.shape
        T = total_nodes // 3

        text_nodes = fused_features[:, :T, :]       # [batch, T, hidden_dim]
        audio_nodes = fused_features[:, T:2*T, :]   # [batch, T, hidden_dim]
        video_nodes = fused_features[:, 2*T:, :]    # [batch, T, hidden_dim]

        # Masked pooling: 只对有效帧进行平均
        if masks is not None:
            # 扩展 mask
            mask_expanded = masks.unsqueeze(-1).float()  # [batch, T, 1]

            # 计算有效帧数
            valid_counts = masks.sum(dim=1, keepdim=True).float()  # [batch, 1]

            # Masked sum + 平均
            text_pooled = (text_nodes * mask_expanded).sum(dim=1) / valid_counts
            audio_pooled = (audio_nodes * mask_expanded).sum(dim=1) / valid_counts
            video_pooled = (video_nodes * mask_expanded).sum(dim=1) / valid_counts
        else:
            # 普通平均池化
            text_pooled = text_nodes.mean(dim=1)    # [batch, hidden_dim]
            audio_pooled = audio_nodes.mean(dim=1)
            video_pooled = video_nodes.mean(dim=1)

        # 5. 拼接并分类
        multimodal_feature = torch.cat(
            [text_pooled, audio_pooled, video_pooled],
            dim=1
        )  # [batch, hidden_dim * 3]

        logits = self.classifier(multimodal_feature)  # [batch, num_classes]

        # 准备输出
        output = {
            'logits': logits,
            'H': H,
            'text_pooled': text_pooled,
            'audio_pooled': audio_pooled,
            'video_pooled': video_pooled
        }

        # 6. 计算损失（如果提供标签）
        if labels is not None:
            # 分类损失
            cls_loss = F.cross_entropy(logits, labels)
            output['cls_loss'] = cls_loss

            # 对比学习损失
            if self.use_contrastive:
                contrastive_loss = self.contrastive(multimodal_feature, labels)
                output['contrastive_loss'] = contrastive_loss
            else:
                contrastive_loss = torch.tensor(0.0).to(logits.device)
                output['contrastive_loss'] = contrastive_loss

            # L2 正则化
            l2_reg = sum(p.pow(2).sum() for p in self.parameters())
            output['l2_reg'] = l2_reg

            # 总损失
            total_loss = (
                cls_loss +
                self.contrastive_weight * contrastive_loss +
                self.l2_reg_weight * l2_reg
            )
            output['loss'] = total_loss

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


class HypergraphEmotionClassifier(nn.Module):
    """
    超图情感分类器 - 简化接口

    方便与 DataLoader 集成
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
            'num_hyperedges': 64,
            'num_conv_layers': 2,
            'bottleneck_dim': 128,
            'dropout': 0.1,
            'hyperedge_drop_rate': 0.2,
            'use_contrastive': True,
            'use_bottleneck': True,
            'contrastive_weight': 0.1
        }

        if config is not None:
            default_config.update(config)

        self.config = default_config

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
            batch: 来自 DataLoader 的批次数据
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
    model = HypergraphEmotionClassifier(
        feature_dims=feature_dims,
        num_classes=2
    )

    # 前向传播
    output = model(batch)

    print("输出:")
    print(f"  logits: {output['logits'].shape}")
    print(f"  loss: {output['loss'].item():.4f}")
    print(f"  cls_loss: {output['cls_loss'].item():.4f}")
    print(f"  contrastive_loss: {output['contrastive_loss'].item():.4f}")

    # 预测
    predictions = model.predict(batch)
    print(f"\n预测: {predictions}")
    print(f"真实: {batch['labels']}")
