"""
多模态融合网络模块

包含：
1. 单模态编码器 (UnimodalEncoder)
2. 多模态超图融合层 (MultimodalHypergraphLayer)

超图结构：
- 节点：3N个节点，每个样本有3个模态节点
- 超边：N+3条
  * N条样本内超边（权重=1）：连接同一样本的3个模态
  * 3条模态内超边（权重=cos相似度）：连接同一模态的所有样本
- 融合：超图卷积，最终取样本内超边特征作为多模态表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class UnimodalEncoder(nn.Module):
    """
    单模态特征编码器

    将原始特征映射到统一的嵌入空间
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim]
        Returns:
            encoded: [batch_size, output_dim]
        """
        return self.encoder(x)


class MultimodalHypergraphLayer(nn.Module):
    """
    多模态超图融合层

    超图结构：
    - 节点：3N个 (N个样本 × 3个模态)
    - 超边：N+3条
      1. N条样本内超边：每个样本的3个模态用一条超边连接，权重=1
      2. 3条模态内超边：每个模态内所有样本用一条超边连接，权重=余弦相似度

    超图卷积公式：
    X^(l+1) = σ(D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X^(l) Θ)

    其中：
    - H: 关联矩阵 [num_nodes, num_edges]
    - W: 超边权重对角矩阵 [num_edges, num_edges]
    - D_v: 节点度对角矩阵
    - D_e: 超边度对角矩阵
    - Θ: 可学习参数
    """

    def __init__(
        self,
        text_dim: int = 256,
        audio_dim: int = 256,
        video_dim: int = 256,
        hidden_dim: int = 256,
        num_hyperedges: int = 64,  # 保留参数但不使用
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        assert text_dim == audio_dim == video_dim, "所有模态维度必须相同"

        self.feature_dim = text_dim
        self.num_layers = num_layers
        self.num_modalities = 3

        # 超图卷积层（每层学习不同的变换）
        self.hypergraph_convs = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.feature_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers)
        ])

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def compute_cosine_similarity_weights(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算特征间的余弦相似度权重矩阵，使用softmax转换为概率形式

        Args:
            features: [batch_size, feature_dim]

        Returns:
            weights: [batch_size, batch_size] softmax归一化的相似度概率
        """
        # L2归一化
        features_norm = F.normalize(features, p=2, dim=1)

        # 余弦相似度矩阵
        similarity = torch.mm(features_norm, features_norm.t())  # [N, N]

        # 使用softmax转换为概率形式（沿行方向归一化）
        # 每行表示一个节点与其他节点的连接概率分布
        weights = F.softmax(similarity, dim=1)  # [N, N]

        return weights

    def hypergraph_convolution(
        self,
        node_features: torch.Tensor,
        incidence_matrix: torch.Tensor,
        edge_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        超图卷积操作

        Args:
            node_features: [num_nodes, feature_dim]
            incidence_matrix: [num_nodes, num_edges]
            edge_weights: [num_edges, num_edges] 对角矩阵

        Returns:
            updated_features: [num_nodes, feature_dim]
        """
        # 计算节点度 D_v
        D_v = torch.diag(incidence_matrix.sum(dim=1))  # [num_nodes, num_nodes]
        D_v_inv_sqrt = torch.diag(1.0 / (torch.diagonal(D_v).sqrt() + 1e-10))

        # 计算超边度 D_e
        D_e = torch.diag(incidence_matrix.sum(dim=0))  # [num_edges, num_edges]
        D_e_inv = torch.diag(1.0 / (torch.diagonal(D_e) + 1e-10))

        # 超图卷积：D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X
        x = torch.mm(D_v_inv_sqrt, node_features)  # [num_nodes, feature_dim]
        x = torch.mm(incidence_matrix, torch.mm(edge_weights, torch.mm(D_e_inv, incidence_matrix.t())))  # [num_nodes, num_nodes]
        x = torch.mm(x, torch.mm(D_v_inv_sqrt, node_features))  # [num_nodes, feature_dim]

        return x

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor
    ) -> torch.Tensor:
        """
        超图融合前向传播

        Args:
            text_features: [batch_size, text_dim]
            audio_features: [batch_size, audio_dim]
            video_features: [batch_size, video_dim]

        Returns:
            fused_features: [batch_size, hidden_dim] 融合后的多模态特征
        """
        batch_size = text_features.size(0)
        num_nodes = batch_size * self.num_modalities  # 3N个节点
        num_edges = batch_size + self.num_modalities  # N+3条超边

        # ====================================================================
        # 1. 构建节点特征矩阵 [3N, feature_dim]
        # ====================================================================
        # 节点顺序：[text_0, audio_0, video_0, text_1, audio_1, video_1, ...]
        node_features_list = []
        for i in range(batch_size):
            node_features_list.append(text_features[i])
            node_features_list.append(audio_features[i])
            node_features_list.append(video_features[i])

        node_features = torch.stack(node_features_list, dim=0)  # [3N, feature_dim]

        # ====================================================================
        # 2. 构建关联矩阵 H [3N, N+3]
        # ====================================================================
        device = text_features.device
        H = torch.zeros(num_nodes, num_edges, device=device)

        # 2.1 样本内超边（前N条）：连接每个样本的3个模态
        for i in range(batch_size):
            edge_idx = i
            # 样本i的3个模态节点
            text_node = i * 3
            audio_node = i * 3 + 1
            video_node = i * 3 + 2

            H[text_node, edge_idx] = 1.0
            H[audio_node, edge_idx] = 1.0
            H[video_node, edge_idx] = 1.0

        # 2.2 模态内超边（后3条）：连接同一模态的所有样本
        # Text模态超边
        text_edge_idx = batch_size
        text_weights = self.compute_cosine_similarity_weights(text_features)  # [N, N]
        for i in range(batch_size):
            text_node = i * 3
            for j in range(batch_size):
                # 使用余弦相似度作为连接权重
                H[text_node, text_edge_idx] = text_weights[i, j]

        # Audio模态超边
        audio_edge_idx = batch_size + 1
        audio_weights = self.compute_cosine_similarity_weights(audio_features)  # [N, N]
        for i in range(batch_size):
            audio_node = i * 3 + 1
            for j in range(batch_size):
                H[audio_node, audio_edge_idx] = audio_weights[i, j]

        # Video模态超边
        video_edge_idx = batch_size + 2
        video_weights = self.compute_cosine_similarity_weights(video_features)  # [N, N]
        for i in range(batch_size):
            video_node = i * 3 + 2
            for j in range(batch_size):
                H[video_node, video_edge_idx] = video_weights[i, j]

        # ====================================================================
        # 3. 构建超边权重矩阵 W [N+3, N+3]
        # ====================================================================
        W = torch.eye(num_edges, device=device)
        # 样本内超边权重为1（前N条）
        for i in range(batch_size):
            W[i, i] = 1.0
        # 模态内超边权重为1（后3条，相似度已经在H中体现）
        for i in range(self.num_modalities):
            W[batch_size + i, batch_size + i] = 1.0

        # ====================================================================
        # 4. 多层超图卷积
        # ====================================================================
        x = node_features

        for layer_idx, conv_layer in enumerate(self.hypergraph_convs):
            # 超图卷积
            x_conv = self.hypergraph_convolution(x, H, W)

            # 通过可学习变换
            x_transformed = conv_layer(x_conv)

            # 残差连接（从第2层开始）
            if layer_idx > 0:
                x = x_transformed + x
            else:
                x = x_transformed

        # ====================================================================
        # 5. 提取样本内超边特征作为最终多模态表示
        # ====================================================================
        # 每个样本的多模态表示 = 该样本3个模态节点的平均
        fused_features_list = []
        for i in range(batch_size):
            text_node = i * 3
            audio_node = i * 3 + 1
            video_node = i * 3 + 2

            # 样本i的多模态表示 = 3个模态节点特征的平均
            sample_feature = (x[text_node] + x[audio_node] + x[video_node]) / 3.0
            fused_features_list.append(sample_feature)

        fused_features = torch.stack(fused_features_list, dim=0)  # [batch_size, hidden_dim]

        # 输出投影
        output = self.output_proj(fused_features)

        return output


if __name__ == "__main__":
    # 测试超图融合
    print("测试多模态超图融合...")

    batch_size = 4
    feature_dim = 256

    # 创建测试数据
    text = torch.randn(batch_size, feature_dim)
    audio = torch.randn(batch_size, feature_dim)
    video = torch.randn(batch_size, feature_dim)

    # 创建超图融合层
    hypergraph = MultimodalHypergraphLayer(
        text_dim=feature_dim,
        audio_dim=feature_dim,
        video_dim=feature_dim,
        hidden_dim=256,
        num_layers=2
    )

    # 前向传播
    fused = hypergraph(text, audio, video)

    print(f"\n输入:")
    print(f"  Text: {text.shape}")
    print(f"  Audio: {audio.shape}")
    print(f"  Video: {video.shape}")
    print(f"\n输出:")
    print(f"  Fused: {fused.shape}")

    print(f"\n超图结构:")
    print(f"  节点数: {batch_size * 3} (3个模态 × {batch_size}个样本)")
    print(f"  超边数: {batch_size + 3} ({batch_size}条样本内超边 + 3条模态内超边)")
    print(f"  - 样本内超边: 每个样本的3个模态用权重=1的超边连接")
    print(f"  - 模态内超边: 同一模态的所有样本用余弦相似度加权的超边连接")

    print("\n✓ 超图融合测试通过!")
