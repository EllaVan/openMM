"""
基于超图结构的多模态融合预测网络
结合超图卷积神经网络和图对比学习

超图结构:
- 节点: M*N 个 (N个样本, M=3个模态: text/video/audio)
- 超边: M+N 条
  - N条: 每个样本内的所有单模态特征之间有一条超边连接
  - M条: 不同样本同一模态之间，满足K最近邻时有超边连接
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import numpy as np
from sklearn.neighbors import NearestNeighbors


class HypergraphConstructor:
    """
    超图构建器
    构建关联矩阵 H: (M*N) × (M+N)
    - 行: M*N个节点 (每个样本有M个模态节点)
    - 列: M+N条超边 (N条样本内超边 + M条模态间超边)
    """

    def __init__(self, num_samples: int, num_modalities: int = 3, k_neighbors: int = 5):
        """
        Args:
            num_samples (int): 样本数量 N
            num_modalities (int): 模态数量 M (默认3: text/video/audio)
            k_neighbors (int): K最近邻的K值
        """
        self.N = num_samples
        self.M = num_modalities
        self.k = k_neighbors
        self.num_nodes = self.M * self.N  # 节点数量
        self.num_hyperedges = self.M + self.N  # 超边数量

    def construct_incidence_matrix(
        self,
        features_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        构建关联矩阵 H

        Args:
            features_list: 长度为M的列表，每个元素形状为(N, D_m)
                          表示M个模态的特征，每个模态有N个样本

        Returns:
            H: 关联矩阵，形状为 (M*N, M+N)
        """
        device = features_list[0].device
        H = torch.zeros(self.num_nodes, self.num_hyperedges, device=device)

        # 1. 构建样本内超边 (N条超边)
        # 每个样本的M个模态节点连接到同一条超边
        for sample_idx in range(self.N):
            hyperedge_idx = sample_idx  # 前N条超边对应样本内连接
            for modality_idx in range(self.M):
                node_idx = sample_idx * self.M + modality_idx
                H[node_idx, hyperedge_idx] = 1

        # 2. 构建模态间超边 (M条超边)
        # 对于每个模态，使用K最近邻连接不同样本
        for modality_idx in range(self.M):
            hyperedge_idx = self.N + modality_idx  # 后M条超边对应模态间连接

            # 获取当前模态的所有样本特征
            modality_features = features_list[modality_idx]  # (N, D_m)

            # 计算K最近邻
            knn_edges = self._compute_knn(modality_features)

            # 为每个K最近邻组创建连接
            for sample_idx in range(self.N):
                node_idx = sample_idx * self.M + modality_idx
                H[node_idx, hyperedge_idx] = 1

                # 连接K最近邻
                neighbors = knn_edges[sample_idx]
                for neighbor_idx in neighbors:
                    neighbor_node_idx = neighbor_idx * self.M + modality_idx
                    H[neighbor_node_idx, hyperedge_idx] = 1

        return H

    def _compute_knn(self, features: torch.Tensor) -> np.ndarray:
        """
        计算K最近邻

        Args:
            features: 特征矩阵 (N, D)

        Returns:
            knn_indices: K最近邻索引 (N, k)
        """
        features_np = features.detach().cpu().numpy()

        # 使用sklearn计算K最近邻
        nbrs = NearestNeighbors(n_neighbors=self.k + 1, algorithm='auto').fit(features_np)
        distances, indices = nbrs.kneighbors(features_np)

        # 排除自己，返回K个最近邻
        knn_indices = indices[:, 1:]

        return knn_indices


class HypergraphConvolution(nn.Module):
    """
    超图卷积层

    实现公式: X' = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X Θ
    其中:
    - H: 关联矩阵 (num_nodes, num_hyperedges)
    - D_v: 节点度矩阵 (对角矩阵)
    - D_e: 超边度矩阵 (对角矩阵)
    - W: 超边权重矩阵 (对角矩阵)
    - Θ: 可学习的权重矩阵
    """

    def __init__(self, in_features: int, out_features: int, dropout: float = 0.5):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            dropout: Dropout概率
        """
        super(HypergraphConvolution, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 可学习的权重矩阵
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))

        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            X: 节点特征矩阵 (num_nodes, in_features)
            H: 关联矩阵 (num_nodes, num_hyperedges)

        Returns:
            X_out: 输出特征矩阵 (num_nodes, out_features)
        """
        # 计算节点度矩阵 D_v
        D_v = torch.sum(H, dim=1)  # (num_nodes,)
        D_v_inv_sqrt = torch.pow(D_v + 1e-10, -0.5)  # 避免除零
        D_v_inv_sqrt = torch.diag(D_v_inv_sqrt)  # (num_nodes, num_nodes)

        # 计算超边度矩阵 D_e
        D_e = torch.sum(H, dim=0)  # (num_hyperedges,)
        D_e_inv = torch.pow(D_e + 1e-10, -1.0)
        D_e_inv = torch.diag(D_e_inv)  # (num_hyperedges, num_hyperedges)

        # 超边权重矩阵 W (简化为单位矩阵)
        W = torch.eye(H.shape[1], device=H.device)

        # 超图拉普拉斯矩阵: L = D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2}
        # 简化计算: X' = L X Θ
        HT = H.t()  # (num_hyperedges, num_nodes)

        # 计算: D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X
        temp = torch.mm(HT, torch.mm(D_v_inv_sqrt, X))  # H^T D_v^{-1/2} X
        temp = torch.mm(D_e_inv, temp)  # D_e^{-1} H^T D_v^{-1/2} X
        temp = torch.mm(W, temp)  # W D_e^{-1} H^T D_v^{-1/2} X
        temp = torch.mm(H, temp)  # H W D_e^{-1} H^T D_v^{-1/2} X
        temp = torch.mm(D_v_inv_sqrt, temp)  # D_v^{-1/2} H W D_e^{-1} H^T D_v^{-1/2} X

        # 应用可学习权重
        output = torch.mm(temp, self.weight) + self.bias

        # Dropout
        output = self.dropout(output)

        return output


class ContrastiveLearning(nn.Module):
    """
    图对比学习模块

    使用监督对比学习损失 (Supervised Contrastive Learning)
    同类样本特征拉近，异类样本特征推远
    """

    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: 温度参数，控制对比学习的强度
        """
        super(ContrastiveLearning, self).__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算监督对比学习损失

        Args:
            features: 特征向量 (batch_size, feature_dim)
            labels: 标签 (batch_size,)
            mask: 可选的mask矩阵 (batch_size, batch_size)

        Returns:
            loss: 对比学习损失
        """
        device = features.device
        batch_size = features.shape[0]

        # L2归一化
        features = F.normalize(features, dim=1)

        # 计算相似度矩阵
        similarity_matrix = torch.mm(features, features.t())  # (batch_size, batch_size)

        # 创建标签mask: 同类为1，异类为0
        labels = labels.contiguous().view(-1, 1)
        mask_label = torch.eq(labels, labels.t()).float().to(device)  # (batch_size, batch_size)

        # 排除自己
        mask_self = torch.eye(batch_size, device=device)
        mask_label = mask_label * (1 - mask_self)

        # 应用温度参数
        similarity_matrix = similarity_matrix / self.temperature

        # 计算exp(similarity)
        exp_sim = torch.exp(similarity_matrix)

        # 排除对角线（自己）
        exp_sim = exp_sim * (1 - mask_self)

        # 计算分母：所有负样本对的和
        sum_exp_sim = exp_sim.sum(dim=1, keepdim=True)

        # 计算对比学习损失
        # loss = -log(exp(sim_pos) / sum(exp(sim)))
        log_prob = similarity_matrix - torch.log(sum_exp_sim + 1e-10)

        # 只计算正样本对的损失
        mask_label_sum = mask_label.sum(dim=1)
        mask_label_sum = torch.clamp(mask_label_sum, min=1.0)  # 避免除零

        loss = -(mask_label * log_prob).sum(dim=1) / mask_label_sum
        loss = loss.mean()

        return loss


class MultimodalHypergraphNetwork(nn.Module):
    """
    基于超图结构的多模态融合预测网络

    结合超图卷积神经网络和图对比学习进行监督分类
    """

    def __init__(
        self,
        feature_dims: List[int],  # 每个模态的特征维度 [D_text, D_video, D_audio]
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_classes: int = 6,
        num_hgcn_layers: int = 2,
        k_neighbors: int = 5,
        dropout: float = 0.5,
        temperature: float = 0.07
    ):
        """
        Args:
            feature_dims: 每个模态的特征维度列表
            hidden_dim: 隐藏层维度
            output_dim: 输出嵌入维度
            num_classes: 分类类别数
            num_hgcn_layers: 超图卷积层数量
            k_neighbors: K最近邻的K值
            dropout: Dropout概率
            temperature: 对比学习温度参数
        """
        super(MultimodalHypergraphNetwork, self).__init__()

        self.num_modalities = len(feature_dims)
        self.feature_dims = feature_dims
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.k_neighbors = k_neighbors

        # 特征投影层：将不同模态的特征投影到统一空间
        self.feature_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for dim in feature_dims
        ])

        # 超图卷积层
        self.hgcn_layers = nn.ModuleList()
        for i in range(num_hgcn_layers):
            in_dim = hidden_dim if i == 0 else output_dim
            self.hgcn_layers.append(
                HypergraphConvolution(in_dim, output_dim, dropout)
            )

        # 激活函数
        self.relu = nn.ReLU()

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        # 对比学习模块
        self.contrastive_learning = ContrastiveLearning(temperature)

        # 超图构建器（会在forward中初始化）
        self.hypergraph_constructor = None

    def forward(
        self,
        features_list: List[torch.Tensor],  # [text_feat, video_feat, audio_feat]
        labels: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> dict:
        """
        前向传播

        Args:
            features_list: 长度为M的列表，每个元素形状为(N, D_m)
            labels: 标签 (N,)，训练时需要提供
            return_embeddings: 是否返回节点嵌入

        Returns:
            outputs: 字典，包含分类结果、损失等
        """
        batch_size = features_list[0].shape[0]
        device = features_list[0].device

        # 1. 特征投影到统一空间
        projected_features = []
        for i, features in enumerate(features_list):
            proj_feat = self.feature_projections[i](features)  # (N, hidden_dim)
            projected_features.append(proj_feat)

        # 2. 构建超图
        if self.hypergraph_constructor is None or self.hypergraph_constructor.N != batch_size:
            self.hypergraph_constructor = HypergraphConstructor(
                num_samples=batch_size,
                num_modalities=self.num_modalities,
                k_neighbors=self.k_neighbors
            )

        H = self.hypergraph_constructor.construct_incidence_matrix(projected_features)

        # 3. 构建节点特征矩阵 X: (M*N, hidden_dim)
        # 将所有模态的特征堆叠成节点特征
        X = torch.cat([feat for feat in projected_features], dim=0)  # (M*N, hidden_dim)

        # 4. 超图卷积
        for i, hgcn_layer in enumerate(self.hgcn_layers):
            X = hgcn_layer(X, H)
            if i < len(self.hgcn_layers) - 1:
                X = self.relu(X)

        # 5. 节点特征聚合
        # 将每个样本的M个模态节点聚合成一个样本表示
        X_reshaped = X.view(batch_size, self.num_modalities, self.output_dim)

        # 使用平均池化聚合多模态特征
        sample_embeddings = torch.mean(X_reshaped, dim=1)  # (N, output_dim)

        # 6. 分类
        logits = self.classifier(sample_embeddings)  # (N, num_classes)

        # 7. 计算损失
        outputs = {'logits': logits}

        if labels is not None:
            # 分类损失
            classification_loss = F.cross_entropy(logits, labels)

            # 对比学习损失
            contrastive_loss = self.contrastive_learning(sample_embeddings, labels)

            # 总损失
            total_loss = classification_loss + 0.5 * contrastive_loss

            outputs['loss'] = total_loss
            outputs['classification_loss'] = classification_loss
            outputs['contrastive_loss'] = contrastive_loss

        if return_embeddings:
            outputs['embeddings'] = sample_embeddings
            outputs['node_features'] = X

        return outputs

    def predict(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        预测

        Args:
            features_list: 特征列表

        Returns:
            predictions: 预测类别 (N,)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(features_list)
            predictions = torch.argmax(outputs['logits'], dim=1)
        return predictions
