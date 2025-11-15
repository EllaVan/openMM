"""
样本级别超图模块

超图构建：
- 节点：3N个（N个样本 × 3个模态）
- 超边：N+3条
  * N条样本内超边：连接同一样本的3个不同模态
  * 3条模态内超边：连接所有样本的同一模态，权重为样本间cos相似度
- 样本表示：聚合每个样本的3个模态特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class SampleLevelHypergraph(nn.Module):
    """
    样本级别超图网络

    构建batch内样本的超图，节点为每个样本的每个模态特征
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int,
        num_conv_layers: int = 2,
        dropout: float = 0.1,
        use_edge_weights: bool = True,
        similarity_temperature: float = 1.0
    ):
        """
        Args:
            text_dim: 文本特征维度
            audio_dim: 音频特征维度
            video_dim: 视频特征维度
            hidden_dim: 隐藏层维度
            num_conv_layers: 超图卷积层数
            dropout: Dropout率
            use_edge_weights: 是否使用边权重
            similarity_temperature: 相似度温度参数
        """
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_conv_layers = num_conv_layers
        self.use_edge_weights = use_edge_weights
        self.temperature = similarity_temperature

        # 将三个模态投影到统一维度
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        # 超图卷积层
        self.conv_layers = nn.ModuleList([
            SampleHypergraphConv(
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(num_conv_layers)
        ])

        # 样本聚合层
        self.sample_aggregator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def build_hypergraph(
        self,
        nodes: torch.Tensor,
        batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建超图关联矩阵和权重

        Args:
            nodes: 节点特征 [3N, hidden_dim]
            batch_size: 批次大小 N

        Returns:
            H: 关联矩阵 [3N, N+3]
            W: 超边权重 [N+3]
        """
        num_nodes = 3 * batch_size
        num_hyperedges = batch_size + 3

        # 初始化关联矩阵
        H = torch.zeros(num_nodes, num_hyperedges, device=nodes.device)

        # 1. 构建样本内超边 (前N条超边)
        # 每条超边连接同一样本的3个模态
        for i in range(batch_size):
            text_idx = i                    # 文本节点索引
            audio_idx = batch_size + i      # 音频节点索引
            video_idx = 2 * batch_size + i  # 视频节点索引
            hyperedge_idx = i               # 超边索引

            H[text_idx, hyperedge_idx] = 1.0
            H[audio_idx, hyperedge_idx] = 1.0
            H[video_idx, hyperedge_idx] = 1.0

        # 2. 构建模态内超边 (后3条超边)
        # 每条超边连接所有样本的同一模态

        # 提取各模态的节点特征
        text_nodes = nodes[:batch_size]              # [N, hidden_dim]
        audio_nodes = nodes[batch_size:2*batch_size] # [N, hidden_dim]
        video_nodes = nodes[2*batch_size:]           # [N, hidden_dim]

        modality_nodes = [text_nodes, audio_nodes, video_nodes]

        for modality_idx, modal_nodes in enumerate(modality_nodes):
            hyperedge_idx = batch_size + modality_idx

            if self.use_edge_weights:
                # 计算样本间的cos相似度
                # 归一化特征
                modal_nodes_norm = F.normalize(modal_nodes, p=2, dim=1)

                # 计算相似度矩阵 [N, N]
                similarity = torch.mm(modal_nodes_norm, modal_nodes_norm.t())

                # 对每个样本，计算其与所有样本的平均相似度作为权重
                weights = similarity.mean(dim=1) / self.temperature
                weights = torch.softmax(weights, dim=0)
            else:
                # 均匀权重
                weights = torch.ones(batch_size, device=nodes.device) / batch_size

            # 设置超边连接权重
            for sample_idx in range(batch_size):
                node_idx = modality_idx * batch_size + sample_idx
                H[node_idx, hyperedge_idx] = weights[sample_idx]

        # 3. 计算超边权重
        if self.use_edge_weights:
            # 样本内超边权重为1
            W = torch.ones(num_hyperedges, device=nodes.device)
            # 可以根据需要调整模态内超边的权重
            W[batch_size:] = 1.0
        else:
            W = torch.ones(num_hyperedges, device=nodes.device)

        return H, W

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        return_H: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播

        Args:
            text_features: [batch_size, text_dim] (已pooling的特征)
            audio_features: [batch_size, audio_dim]
            video_features: [batch_size, video_dim]
            return_H: 是否返回关联矩阵

        Returns:
            dict: {
                'sample_features': 样本表示 [batch_size, hidden_dim],
                'text_features': 文本节点特征 [batch_size, hidden_dim],
                'audio_features': 音频节点特征 [batch_size, hidden_dim],
                'video_features': 视频节点特征 [batch_size, hidden_dim],
                'H': 关联矩阵 (可选) [3N, N+3]
            }
        """
        batch_size = text_features.shape[0]

        # 1. 投影到统一维度
        text_proj = self.text_proj(text_features)    # [N, hidden_dim]
        audio_proj = self.audio_proj(audio_features)  # [N, hidden_dim]
        video_proj = self.video_proj(video_features)  # [N, hidden_dim]

        # 2. 拼接所有节点: [文本节点 | 音频节点 | 视频节点]
        nodes = torch.cat([text_proj, audio_proj, video_proj], dim=0)
        # [3N, hidden_dim]

        # 3. 构建超图
        H, W = self.build_hypergraph(nodes, batch_size)

        # 4. 超图卷积
        nodes_out = nodes
        for conv_layer in self.conv_layers:
            nodes_out = conv_layer(nodes_out, H, W)

        # 5. 分离三个模态的节点
        text_out = nodes_out[:batch_size]              # [N, hidden_dim]
        audio_out = nodes_out[batch_size:2*batch_size]  # [N, hidden_dim]
        video_out = nodes_out[2*batch_size:]           # [N, hidden_dim]

        # 6. 聚合每个样本的3个模态特征
        # 拼接三个模态
        sample_concat = torch.cat([text_out, audio_out, video_out], dim=1)
        # [N, hidden_dim * 3]

        # 聚合得到样本表示
        sample_features = self.sample_aggregator(sample_concat)
        # [N, hidden_dim]

        result = {
            'sample_features': sample_features,
            'text_features': text_out,
            'audio_features': audio_out,
            'video_features': video_out
        }

        if return_H:
            result['H'] = H
            result['W'] = W

        return result


class SampleHypergraphConv(nn.Module):
    """
    样本级别超图卷积层

    两阶段传播:
    1. 节点 → 超边 (聚合同一超边的节点特征)
    2. 超边 → 节点 (传播超边特征到节点)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            dropout: Dropout率
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # 可学习的权重矩阵
        self.theta = nn.Linear(in_dim, out_dim, bias=True)

        # 批归一化
        self.bn = nn.BatchNorm1d(out_dim)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(
        self,
        nodes: torch.Tensor,
        H: torch.Tensor,
        W: torch.Tensor
    ) -> torch.Tensor:
        """
        超图卷积

        Args:
            nodes: 节点特征 [num_nodes, in_dim]
            H: 关联矩阵 [num_nodes, num_hyperedges]
            W: 超边权重 [num_hyperedges]

        Returns:
            nodes_out: 更新后的节点特征 [num_nodes, out_dim]
        """
        num_nodes, num_hyperedges = H.shape

        # 计算度矩阵
        # D_n: 节点度 [num_nodes]
        D_n = H.sum(dim=1)  # 每个节点连接的超边权重和
        D_n_inv_sqrt = torch.pow(D_n + 1e-10, -0.5)

        # D_e: 超边度 [num_hyperedges]
        D_e = H.sum(dim=0)  # 每条超边连接的节点权重和
        D_e_inv_sqrt = torch.pow(D_e + 1e-10, -0.5)

        # W 作为对角矩阵
        W_diag = torch.diag(W)  # [num_hyperedges, num_hyperedges]

        # 归一化的关联矩阵
        # H_norm = D_n^{-1/2} H W D_e^{-1/2}
        H_norm = H * D_n_inv_sqrt.unsqueeze(1)  # [num_nodes, num_hyperedges]
        H_norm = torch.mm(H_norm, W_diag)        # [num_nodes, num_hyperedges]
        H_norm = H_norm * D_e_inv_sqrt.unsqueeze(0)  # [num_nodes, num_hyperedges]

        # 超图拉普拉斯卷积
        # N' = H_norm H_norm^T N
        nodes_agg = torch.mm(H_norm, H_norm.t())  # [num_nodes, num_nodes]
        nodes_agg = torch.mm(nodes_agg, nodes)     # [num_nodes, in_dim]

        # 应用可学习参数
        nodes_out = self.theta(nodes_agg)  # [num_nodes, out_dim]

        # 批归一化
        nodes_out = self.bn(nodes_out)

        # Dropout
        nodes_out = self.dropout(nodes_out)

        # 激活函数
        nodes_out = self.activation(nodes_out)

        # 残差连接
        if self.in_dim == self.out_dim:
            nodes_out = nodes_out + nodes

        return nodes_out


if __name__ == "__main__":
    # 测试样本级别超图
    batch_size = 8
    text_dim, audio_dim, video_dim = 768, 768, 768
    hidden_dim = 256

    # 创建测试数据 (已经pooling后的样本特征)
    text_features = torch.randn(batch_size, text_dim)
    audio_features = torch.randn(batch_size, audio_dim)
    video_features = torch.randn(batch_size, video_dim)

    # 创建超图模型
    hypergraph = SampleLevelHypergraph(
        text_dim=text_dim,
        audio_dim=audio_dim,
        video_dim=video_dim,
        hidden_dim=hidden_dim,
        num_conv_layers=2
    )

    # 前向传播
    output = hypergraph(
        text_features,
        audio_features,
        video_features,
        return_H=True
    )

    print("=" * 70)
    print("样本级别超图测试")
    print("=" * 70)
    print(f"\n输入:")
    print(f"  Batch size: {batch_size}")
    print(f"  Text features: {text_features.shape}")
    print(f"  Audio features: {audio_features.shape}")
    print(f"  Video features: {video_features.shape}")

    print(f"\n超图结构:")
    print(f"  节点数: {3 * batch_size} (3个模态 × {batch_size}个样本)")
    print(f"  超边数: {batch_size + 3} ({batch_size}条样本内超边 + 3条模态内超边)")
    print(f"  关联矩阵 H: {output['H'].shape}")
    print(f"  超边权重 W: {output['W'].shape}")

    print(f"\n输出:")
    print(f"  Sample features: {output['sample_features'].shape}")
    print(f"  Text features: {output['text_features'].shape}")
    print(f"  Audio features: {output['audio_features'].shape}")
    print(f"  Video features: {output['video_features'].shape}")

    print(f"\n关联矩阵示例 (前5个节点，前8条超边):")
    print(output['H'][:5, :8])

    print(f"\n超边权重:")
    print(f"  样本内超边权重: {output['W'][:batch_size]}")
    print(f"  模态内超边权重: {output['W'][batch_size:]}")

    print("\n" + "=" * 70)
