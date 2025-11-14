"""
超图网络模块 - 支持 Padding + Masking
基于 hyper_graph_fusion_instruct.md 实现

核心功能:
1. 超图初始化 (基于相关性，支持 mask)
2. 超图增强 (随机删除超边)
3. 超图卷积 (两阶段传播，支持 mask)
4. 图对比学习
5. 多模态融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class HypergraphInitializer(nn.Module):
    """
    超图初始化模块 - 支持 Mask

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
        self.W_E = nn.Linear(node_dim, embed_dim, bias=False)

        # 可学习的超边原型
        self.hyperedge_prototypes = nn.Parameter(
            torch.randn(num_hyperedges, node_dim)
        )

    def forward(
        self,
        nodes: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        构建超图连接矩阵

        Args:
            nodes: 节点特征 [batch_size, num_nodes, node_dim]
            mask: 节点有效性 mask [batch_size, num_nodes]
                  True 表示有效节点，False 表示填充节点

        Returns:
            H: 连接矩阵 [batch_size, num_nodes, num_hyperedges]
        """
        batch_size, num_nodes, _ = nodes.shape

        # 节点嵌入
        Z_N = self.W_N(nodes)  # [batch_size, num_nodes, embed_dim]

        # 超边嵌入
        Z_E = self.W_E(self.hyperedge_prototypes)  # [K, embed_dim]
        Z_E = Z_E.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, K, embed_dim]

        # 计算相似度矩阵
        H_hat = torch.bmm(Z_N, Z_E.transpose(1, 2))  # [batch, num_nodes, K]

        # 如果有 mask，将填充位置设为极小值（softmax 后接近 0）
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)  # [batch, num_nodes, 1]
            H_hat = H_hat.masked_fill(~mask_expanded, float('-inf'))

        # Softmax 归一化
        H = F.softmax(H_hat / self.temperature, dim=-1)

        # 将填充位置的值重置为 0
        if mask is not None:
            H = H * mask_expanded.float()

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

        batch_size, num_nodes, num_hyperedges = H.shape

        # 为每个样本的每个超边生成保留 mask
        keep_mask = torch.bernoulli(
            torch.full((batch_size, 1, num_hyperedges), 1 - self.drop_rate)
        ).to(H.device)

        # 应用 mask 并重新归一化
        H_aug = H * keep_mask

        # 重新归一化
        row_sums = H_aug.sum(dim=-1, keepdim=True)
        H_aug = H_aug / (row_sums + 1e-10)

        return H_aug


class HypergraphConvolution(nn.Module):
    """
    超图卷积层 - 支持 Mask

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

        # 超边权重 W
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
        H: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        超图卷积

        Args:
            nodes: 节点特征 [batch_size, num_nodes, in_dim]
            H: 连接矩阵 [batch_size, num_nodes, num_hyperedges]
            mask: 节点有效性 mask [batch_size, num_nodes]

        Returns:
            nodes_out: 更新后的节点特征 [batch_size, num_nodes, out_dim]
        """
        batch_size, num_nodes, _ = nodes.shape

        # 如果有 mask，将填充节点的特征置零
        if mask is not None:
            nodes = nodes * mask.unsqueeze(-1).float()

        # 计算度矩阵
        D_n = H.sum(dim=-1)  # [batch_size, num_nodes]
        D_n_inv_sqrt = torch.pow(D_n + 1e-10, -0.5)

        D_e = H.sum(dim=1)  # [batch_size, num_hyperedges]
        D_e_inv_sqrt = torch.pow(D_e + 1e-10, -0.5)

        # Step 1: 节点到超边聚合
        N_norm = nodes * D_n_inv_sqrt.unsqueeze(-1)
        E = torch.bmm(H.transpose(1, 2), N_norm)  # [batch, num_hyperedges, in_dim]
        E = E * D_e_inv_sqrt.unsqueeze(-1) * self.W

        # Step 2: 超边到节点聚合
        N_prime = torch.bmm(H, E)  # [batch, num_nodes, in_dim]
        N_prime = N_prime * D_n_inv_sqrt.unsqueeze(-1)

        # 如果有 mask，再次确保填充位置为 0
        if mask is not None:
            N_prime = N_prime * mask.unsqueeze(-1).float()

        # 应用可学习参数 θ
        N_out = self.theta(N_prime)

        # 批归一化
        if isinstance(self.bn, nn.BatchNorm1d):
            N_out = N_out.transpose(1, 2)
            N_out = self.bn(N_out)
            N_out = N_out.transpose(1, 2)

        # Dropout
        N_out = self.dropout(N_out)

        # 激活函数
        N_out = self.activation(N_out)

        # 如果有 mask，最后再应用一次
        if mask is not None:
            N_out = N_out * mask.unsqueeze(-1).float()

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

        # 投影头
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

        # 投影
        z = self.projection(features)
        z = F.normalize(z, p=2, dim=1)

        # 计算相似度
        similarity = torch.mm(z, z.t()) / self.temperature

        # 构建标签 mask
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(features.device)

        # 去除对角线
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

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        loss = -mean_log_prob_pos.mean()

        return loss


class BottleneckLayer(nn.Module):
    """
    Bottleneck 层

    降维 → 升维，实现特征压缩和正则化
    """

    def __init__(
        self,
        in_dim: int,
        bottleneck_dim: int,
        out_dim: int,
        dropout: float = 0.1
    ):
        """
        Args:
            in_dim: 输入维度
            bottleneck_dim: Bottleneck 维度
            out_dim: 输出维度
            dropout: Dropout 率
        """
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: 输入 [batch_size, ..., in_dim]

        Returns:
            out: 输出 [batch_size, ..., out_dim]
            bottleneck: Bottleneck 特征 [batch_size, ..., bottleneck_dim]
        """
        bottleneck = self.encoder(x)
        out = self.decoder(bottleneck)

        return out, bottleneck


class MultimodalHypergraphLayer(nn.Module):
    """
    多模态超图层 - 支持 Mask

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
                in_dim=hidden_dim,
                out_dim=hidden_dim,
                use_bn=True,
                dropout=dropout
            )
            for _ in range(num_conv_layers)
        ])

        # 残差连接
        self.use_residual = True

    def forward(
        self,
        text_features: torch.Tensor,
        audio_features: torch.Tensor,
        video_features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_H: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            text_features: [batch_size, T, text_dim]
            audio_features: [batch_size, T, audio_dim]
            video_features: [batch_size, T, video_dim]
            mask: [batch_size, T] - 每个模态的有效帧 mask
            return_H: 是否返回连接矩阵

        Returns:
            dict: {
                'fused': 融合后的特征 [batch, 3T, hidden_dim],
                'text': 文本节点特征 [batch, T, hidden_dim],
                'audio': 音频节点特征 [batch, T, hidden_dim],
                'video': 视频节点特征 [batch, T, hidden_dim],
                'H': 连接矩阵 (可选)
            }
        """
        batch_size, T, _ = text_features.shape

        # 投影到统一维度
        text_proj = self.text_proj(text_features)    # [batch, T, hidden_dim]
        audio_proj = self.audio_proj(audio_features)  # [batch, T, hidden_dim]
        video_proj = self.video_proj(video_features)  # [batch, T, hidden_dim]

        # 拼接节点: [文本节点 | 音频节点 | 视频节点]
        nodes = torch.cat([text_proj, audio_proj, video_proj], dim=1)
        # [batch, 3T, hidden_dim]

        # 扩展 mask 到三个模态
        if mask is not None:
            mask_3x = torch.cat([mask, mask, mask], dim=1)  # [batch, 3T]
        else:
            mask_3x = None

        # 构建超图连接矩阵
        H = self.hypergraph_init(nodes, mask=mask_3x)  # [batch, 3T, K]

        # 超图增强（训练时）
        H_aug = self.hypergraph_aug(H, training=self.training)

        # 超图卷积
        nodes_out = nodes
        for conv_layer in self.conv_layers:
            nodes_updated = conv_layer(nodes_out, H_aug, mask=mask_3x)

            # 残差连接
            if self.use_residual:
                nodes_out = nodes_out + nodes_updated
            else:
                nodes_out = nodes_updated

        # 分离三个模态
        text_out = nodes_out[:, :T, :]
        audio_out = nodes_out[:, T:2*T, :]
        video_out = nodes_out[:, 2*T:, :]

        result = {
            'fused': nodes_out,
            'text': text_out,
            'audio': audio_out,
            'video': video_out
        }

        if return_H:
            result['H'] = H

        return result


if __name__ == "__main__":
    # 测试模块
    batch_size = 4
    T = 50  # 最大帧数
    text_dim, audio_dim, video_dim = 768, 768, 768
    hidden_dim = 256

    # 创建测试数据
    text_features = torch.randn(batch_size, T, text_dim)
    audio_features = torch.randn(batch_size, T, audio_dim)
    video_features = torch.randn(batch_size, T, video_dim)

    # 创建 mask（模拟变长序列）
    num_frames = torch.tensor([30, 40, 50, 25])  # 实际帧数
    masks = torch.zeros(batch_size, T, dtype=torch.bool)
    for i, nf in enumerate(num_frames):
        masks[i, :nf] = True

    # 创建超图层
    hypergraph = MultimodalHypergraphLayer(
        text_dim=text_dim,
        audio_dim=audio_dim,
        video_dim=video_dim,
        hidden_dim=hidden_dim,
        num_hyperedges=64,
        num_conv_layers=2
    )

    # 前向传播
    output = hypergraph(text_features, audio_features, video_features, mask=masks, return_H=True)

    print("输出形状:")
    print(f"  fused: {output['fused'].shape}")
    print(f"  text: {output['text'].shape}")
    print(f"  audio: {output['audio'].shape}")
    print(f"  video: {output['video'].shape}")
    print(f"  H: {output['H'].shape}")

    print(f"\nMask 示例 (第一个样本):")
    print(f"  实际帧数: {num_frames[0]}")
    print(f"  mask 前10帧: {masks[0, :10]}")
