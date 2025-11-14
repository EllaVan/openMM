"""
超图网络模块
基于 hyper_graph_fusion_instruct.md 实现的多模态超图融合网络

核心功能:
1. 超图初始化 (基于相关性)
2. 超图增强 (随机删除超边)
3. 超图卷积 (两阶段传播)
4. 图对比学习
5. 多模态融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import math


class HypergraphInitializer(nn.Module):
    """
    超图初始化模块

    基于相关性构建超图连接矩阵 H
    公式: Ĥ = (W_N · N)(W_E · N)^T
         H = softmax(Ĥ / √d)
    """

    def __init__(
        self,
        node_dim: int,
        num_hyperedges: int,
        embed_dim: int = 256,
        temperature: Optional[float] = None
    ):
        """
        Args:
            node_dim: 节点特征维度
            num_hyperedges: 超边数量 K
            embed_dim: 嵌入维度
            temperature: softmax 温度参数，默认为 sqrt(embed_dim)
        """
        super().__init__()

        self.node_dim = node_dim
        self.num_hyperedges = num_hyperedges
        self.embed_dim = embed_dim
        self.temperature = temperature or math.sqrt(embed_dim)

        # 节点投影矩阵 W_N
        self.W_N = nn.Linear(node_dim, embed_dim, bias=False)

        # 超边投影矩阵 W_E
        # 将节点特征映射到超边嵌入空间
        self.W_E = nn.Linear(node_dim, embed_dim, bias=False)

        # 可学习的超边原型
        self.hyperedge_prototypes = nn.Parameter(
            torch.randn(num_hyperedges, node_dim)
        )

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        """
        构建超图连接矩阵

        Args:
            nodes: 节点特征 [batch_size, num_nodes, node_dim]
                  num_nodes = 3T (文本T步 + 音频T步 + 视频T步)

        Returns:
            H: 连接矩阵 [batch_size, num_nodes, num_hyperedges]
               H[b, i, j] 表示第b个样本的第i个节点属于第j个超边的程度
        """
        batch_size, num_nodes, _ = nodes.shape

        # Step 1: 节点嵌入
        # Z_N = W_N · N
        Z_N = self.W_N(nodes)  # [batch_size, num_nodes, embed_dim]

        # Step 2: 超边嵌入
        # Z_E = W_E · prototypes
        Z_E = self.W_E(self.hyperedge_prototypes)  # [num_hyperedges, embed_dim]
        Z_E = Z_E.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, K, embed_dim]

        # Step 3: 计算相似度矩阵
        # Ĥ = Z_N · Z_E^T
        H_hat = torch.bmm(Z_N, Z_E.transpose(1, 2))  # [batch_size, num_nodes, K]

        # Step 4: 温度缩放 + Softmax 归一化
        # H = softmax(Ĥ / √d)
        H = F.softmax(H_hat / self.temperature, dim=-1)

        return H


class HypergraphAugmentation(nn.Module):
    """
    超图增强模块

    通过随机删除超边来减少冗余，提高模型鲁棒性
    """

    def __init__(self, drop_rate: float = 0.2):
        """
        Args:
            drop_rate: 超边删除率
        """
        super().__init__()
        self.drop_rate = drop_rate

    def forward(self, H: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        增强超图连接矩阵

        Args:
            H: 连接矩阵 [batch_size, num_nodes, num_hyperedges]
            training: 是否在训练模式

        Returns:
            H_aug: 增强后的连接矩阵
        """
        if not training or self.drop_rate == 0:
            return H

        # 随机生成超边保留mask
        # 保留概率 = 1 - drop_rate
        batch_size, num_nodes, num_hyperedges = H.shape

        # 为每个样本的每个超边生成保留mask
        keep_mask = torch.bernoulli(
            torch.full((batch_size, 1, num_hyperedges), 1 - self.drop_rate)
        ).to(H.device)

        # 应用mask并重新归一化
        H_aug = H * keep_mask

        # 重新归一化，确保每行和为1
        row_sums = H_aug.sum(dim=-1, keepdim=True)
        H_aug = H_aug / (row_sums + 1e-10)

        return H_aug


class HypergraphConvolution(nn.Module):
    """
    超图卷积层

    两阶段传播:
    1. 节点 → 超边 (聚合同一超边的节点特征)
    2. 超边 → 节点 (传播超边特征到节点)

    公式: G = D_n^(-1/2) H W D_e^(-1/2) H^T D_n^(-1/2)
         N' = σ(G N θ)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = True,
        dropout: float = 0.1
    ):
        """
        Args:
            in_dim: 输入特征维度
            out_dim: 输出特征维度
            use_bn: 是否使用批归一化
            dropout: Dropout 概率
        """
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        # 可学习的权重矩阵 θ
        self.theta = nn.Linear(in_dim, out_dim, bias=False)

        # 超边权重 W (对角矩阵，每个超边一个权重)
        self.W = nn.Parameter(torch.ones(1))

        # 批归一化
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        self.activation = nn.ReLU()

    def forward(
        self,
        nodes: torch.Tensor,
        H: torch.Tensor
    ) -> torch.Tensor:
        """
        超图卷积

        Args:
            nodes: 节点特征 [batch_size, num_nodes, in_dim]
            H: 连接矩阵 [batch_size, num_nodes, num_hyperedges]

        Returns:
            nodes_out: 更新后的节点特征 [batch_size, num_nodes, out_dim]
        """
        batch_size, num_nodes, _ = nodes.shape
        num_hyperedges = H.shape[-1]

        # 计算度矩阵
        # D_n: 节点度矩阵 (每个节点连接的超边数量)
        D_n = H.sum(dim=-1)  # [batch_size, num_nodes]
        D_n_inv_sqrt = torch.pow(D_n + 1e-10, -0.5)  # 防止除零

        # D_e: 超边度矩阵 (每个超边包含的节点数量)
        D_e = H.sum(dim=1)  # [batch_size, num_hyperedges]
        D_e_inv_sqrt = torch.pow(D_e + 1e-10, -0.5)

        # Step 1: 节点到超边聚合
        # E = D_e^(-1/2) H^T D_n^(-1/2) N
        # 先归一化节点
        N_norm = nodes * D_n_inv_sqrt.unsqueeze(-1)  # [batch, num_nodes, in_dim]

        # 聚合到超边
        E = torch.bmm(H.transpose(1, 2), N_norm)  # [batch, num_hyperedges, in_dim]

        # 归一化超边
        E = E * D_e_inv_sqrt.unsqueeze(-1)  # [batch, num_hyperedges, in_dim]

        # 应用超边权重 W
        E = E * self.W

        # Step 2: 超边到节点聚合
        # N' = D_n^(-1/2) H E
        N_prime = torch.bmm(H, E)  # [batch, num_nodes, in_dim]
        N_prime = N_prime * D_n_inv_sqrt.unsqueeze(-1)

        # 应用可学习参数 θ
        N_out = self.theta(N_prime)  # [batch, num_nodes, out_dim]

        # 批归一化 (需要转换维度)
        if isinstance(self.bn, nn.BatchNorm1d):
            # [batch, num_nodes, out_dim] -> [batch, out_dim, num_nodes]
            N_out = N_out.transpose(1, 2)
            N_out = self.bn(N_out)
            N_out = N_out.transpose(1, 2)

        # Dropout
        N_out = self.dropout(N_out)

        # 激活函数
        N_out = self.activation(N_out)

        return N_out


class GraphContrastiveLearning(nn.Module):
    """
    图对比学习模块

    使用监督对比损失增强模态间的一致性
    """

    def __init__(
        self,
        feature_dim: int,
        projection_dim: int = 128,
        temperature: float = 0.07
    ):
        """
        Args:
            feature_dim: 输入特征维度
            projection_dim: 投影头输出维度
            temperature: 对比学习温度参数
        """
        super().__init__()

        self.temperature = temperature

        # 投影头 (MLP)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        计算监督对比损失

        Args:
            features: 特征 [batch_size, feature_dim]
            labels: 标签 [batch_size]

        Returns:
            loss: 对比学习损失
        """
        batch_size = features.shape[0]

        # 投影到低维空间
        z = self.projection(features)  # [batch_size, projection_dim]

        # L2 归一化
        z = F.normalize(z, p=2, dim=1)

        # 计算相似度矩阵
        similarity = torch.mm(z, z.t()) / self.temperature  # [batch_size, batch_size]

        # 构建标签mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # 去除对角线（自己和自己）
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(features.device),
            0
        )
        mask = mask * logits_mask

        # 计算对比损失
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True))

        # 只对正样本对计算损失
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)

        # 损失 (负的平均log概率)
        loss = -mean_log_prob_pos.mean()

        return loss


class BottleneckLayer(nn.Module):
    """
    Bottleneck 层

    压缩特征维度，去除冗余信息
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int,
        out_dim: int
    ):
        """
        Args:
            in_dim: 输入维度
            bottleneck_dim: 瓶颈维度
            out_dim: 输出维度
        """
        super().__init__()

        self.compress = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(),
            nn.BatchNorm1d(bottleneck_dim)
        )

        self.expand = nn.Sequential(
            nn.Linear(bottleneck_dim, out_dim),
            nn.ReLU(),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, in_dim] 或 [batch_size, seq_len, in_dim]

        Returns:
            out: [batch_size, out_dim] 或 [batch_size, seq_len, out_dim]
        """
        is_3d = len(x.shape) == 3

        if is_3d:
            batch_size, seq_len, in_dim = x.shape
            x = x.reshape(batch_size * seq_len, in_dim)

        # 压缩
        z = self.compress(x)

        # 扩展
        out = self.expand(z)

        if is_3d:
            out = out.reshape(batch_size, seq_len, -1)

        return out


class MultimodalHypergraphLayer(nn.Module):
    """
    多模态超图层

    将三个模态的节点统一建模在一个超图中
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        video_dim: int,
        hidden_dim: int,
        num_hyperedges: int,
        num_conv_layers: int = 2,
        dropout: float = 0.1,
        hyperedge_drop_rate: float = 0.2
    ):
        """
        Args:
            text_dim: 文本特征维度
            audio_dim: 音频特征维度
            video_dim: 视频特征维度
            hidden_dim: 隐藏层维度
            num_hyperedges: 超边数量
            num_conv_layers: 超图卷积层数
            dropout: Dropout 率
            hyperedge_drop_rate: 超边删除率
        """
        super().__init__()

        self.text_dim = text_dim
        self.audio_dim = audio_dim
        self.video_dim = video_dim
        self.hidden_dim = hidden_dim

        # 将三个模态投影到统一维度
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.video_proj = nn.Linear(video_dim, hidden_dim)

        # 超图初始化
        self.hypergraph_init = HypergraphInitializer(
            node_dim=hidden_dim,
            num_hyperedges=num_hyperedges,
            embed_dim=hidden_dim
        )

        # 超图增强
        self.hypergraph_aug = HypergraphAugmentation(
            drop_rate=hyperedge_drop_rate
        )

        # 超图卷积层
        self.conv_layers = nn.ModuleList([
            HypergraphConvolution(
                in_dim=hidden_dim if i == 0 else hidden_dim,
                out_dim=hidden_dim,
                use_bn=True,
                dropout=dropout
            )
            for i in range(num_conv_layers)
        ])

        # 残差连接
        self.use_residual = True

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        return_H: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features: [batch_size, T, text_dim]
            audio_features: [batch_size, T, audio_dim]
            video_features: [batch_size, T, video_dim]
            return_H: 是否返回连接矩阵

        Returns:
            dict: {
                'fused': 融合后的特征 [batch_size, 3T, hidden_dim],
                'H': 连接矩阵 (如果 return_H=True)
            }
        """
        batch_size, T, _ = text_features.shape

        # Step 1: 投影到统一维度
        text_proj = self.text_proj(text_features)  # [batch, T, hidden_dim]
        audio_proj = self.audio_proj(audio_features)
        video_proj = self.video_proj(video_features)

        # Step 2: 拼接所有节点
        # 节点顺序: [文本1...文本T, 音频1...音频T, 视频1...视频T]
        nodes = torch.cat([text_proj, audio_proj, video_proj], dim=1)
        # nodes: [batch_size, 3T, hidden_dim]

        # Step 3: 超图初始化
        H = self.hypergraph_init(nodes)  # [batch, 3T, num_hyperedges]

        # Step 4: 超图增强
        H_aug = self.hypergraph_aug(H, training=self.training)

        # Step 5: 超图卷积传播
        nodes_out = nodes
        for i, conv in enumerate(self.conv_layers):
            nodes_conv = conv(nodes_out, H_aug)

            # 残差连接
            if self.use_residual and i > 0:
                nodes_out = nodes_out + nodes_conv
            else:
                nodes_out = nodes_conv

        result = {'fused': nodes_out}

        if return_H:
            result['H'] = H_aug

        return result


# 导出
__all__ = [
    'HypergraphInitializer',
    'HypergraphAugmentation',
    'HypergraphConvolution',
    'GraphContrastiveLearning',
    'BottleneckLayer',
    'MultimodalHypergraphLayer'
]
