"""
超图多模态融合网络使用示例

展示如何使用超图网络进行多模态情感分类
"""

import torch
import numpy as np
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hypergraph_model import (
    MultimodalHypergraphNetwork,
    HypergraphConstructor,
    HypergraphConvolution,
    ContrastiveLearning
)


def example1_hypergraph_construction():
    """示例1: 超图构建"""
    print("=" * 80)
    print("示例1: 超图构建")
    print("=" * 80)

    # 参数
    num_samples = 10  # N=10
    num_modalities = 3  # M=3 (text/video/audio)
    k_neighbors = 3

    # 创建模拟特征数据
    text_features = torch.randn(num_samples, 768)   # (N, 768)
    video_features = torch.randn(num_samples, 512)  # (N, 512)
    audio_features = torch.randn(num_samples, 256)  # (N, 256)

    features_list = [text_features, video_features, audio_features]

    # 构建超图
    constructor = HypergraphConstructor(
        num_samples=num_samples,
        num_modalities=num_modalities,
        k_neighbors=k_neighbors
    )

    H = constructor.construct_incidence_matrix(features_list)

    print(f"\n关联矩阵 H 的形状: {H.shape}")
    print(f"  节点数量: {H.shape[0]} (M*N = {num_modalities}*{num_samples} = {num_modalities*num_samples})")
    print(f"  超边数量: {H.shape[1]} (M+N = {num_modalities}+{num_samples} = {num_modalities+num_samples})")

    print(f"\n超边类型:")
    print(f"  前{num_samples}条超边: 样本内模态融合超边")
    print(f"  后{num_modalities}条超边: 跨样本模态关联超边 (K={k_neighbors}最近邻)")

    # 检查第一个样本的超边
    print(f"\n第一个样本 (样本0) 的节点连接:")
    sample_0_nodes = [0, 1, 2]  # text节点0, video节点1, audio节点2
    print(f"  节点索引: {sample_0_nodes}")
    print(f"  连接到超边0 (样本内超边): {[H[i, 0].item() for i in sample_0_nodes]}")

    # 检查text模态的跨样本超边
    text_hyperedge_idx = num_samples + 0  # 第一个模态间超边
    print(f"\nText模态的跨样本超边 (超边{text_hyperedge_idx}):")
    text_nodes = list(range(0, num_samples * num_modalities, num_modalities))  # 所有text节点
    print(f"  连接的text节点: {[i for i in text_nodes if H[i, text_hyperedge_idx] > 0]}")

    print("\n" + "=" * 80 + "\n")


def example2_hypergraph_convolution():
    """示例2: 超图卷积"""
    print("=" * 80)
    print("示例2: 超图卷积")
    print("=" * 80)

    # 参数
    num_nodes = 30  # M*N = 3*10
    num_hyperedges = 13  # M+N = 3+10
    in_features = 128
    out_features = 64

    # 创建模拟数据
    X = torch.randn(num_nodes, in_features)
    H = torch.rand(num_nodes, num_hyperedges)
    H = (H > 0.5).float()  # 二值化

    # 创建超图卷积层
    hgcn = HypergraphConvolution(in_features, out_features)

    # 前向传播
    X_out = hgcn(X, H)

    print(f"\n输入特征形状: {X.shape}")
    print(f"关联矩阵形状: {H.shape}")
    print(f"输出特征形状: {X_out.shape}")

    print(f"\n超图卷积参数:")
    print(f"  输入维度: {in_features}")
    print(f"  输出维度: {out_features}")
    print(f"  参数数量: {sum(p.numel() for p in hgcn.parameters())}")

    print("\n" + "=" * 80 + "\n")


def example3_contrastive_learning():
    """示例3: 图对比学习"""
    print("=" * 80)
    print("示例3: 图对比学习")
    print("=" * 80)

    # 参数
    batch_size = 16
    feature_dim = 128
    num_classes = 6

    # 创建模拟数据
    features = torch.randn(batch_size, feature_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    # 创建对比学习模块
    contrastive = ContrastiveLearning(temperature=0.07)

    # 计算损失
    loss = contrastive(features, labels)

    print(f"\n特征形状: {features.shape}")
    print(f"标签形状: {labels.shape}")
    print(f"标签分布: {torch.bincount(labels)}")
    print(f"\n对比学习损失: {loss.item():.4f}")

    # 计算同类样本对数量
    labels_expand = labels.unsqueeze(1)
    same_label_mask = (labels_expand == labels_expand.t()).float()
    same_label_mask.fill_diagonal_(0)  # 排除自己
    num_positive_pairs = same_label_mask.sum().item() / 2  # 除以2因为对称

    print(f"同类样本对数量: {int(num_positive_pairs)}")

    print("\n" + "=" * 80 + "\n")


def example4_full_network():
    """示例4: 完整的多模态超图网络"""
    print("=" * 80)
    print("示例4: 完整的多模态超图网络")
    print("=" * 80)

    # 参数
    batch_size = 8
    text_dim = 768
    video_dim = 512
    audio_dim = 256
    num_classes = 6

    # 创建模拟数据
    text_features = torch.randn(batch_size, text_dim)
    video_features = torch.randn(batch_size, video_dim)
    audio_features = torch.randn(batch_size, audio_dim)
    labels = torch.randint(0, num_classes, (batch_size,))

    features_list = [text_features, video_features, audio_features]

    # 创建模型
    model = MultimodalHypergraphNetwork(
        feature_dims=[text_dim, video_dim, audio_dim],
        hidden_dim=256,
        output_dim=128,
        num_classes=num_classes,
        num_hgcn_layers=2,
        k_neighbors=3,
        dropout=0.5,
        temperature=0.07
    )

    print(f"\n模型参数数量: {sum(p.numel() for p in model.parameters())}")

    # 前向传播（训练模式）
    print("\n训练模式（带标签）:")
    outputs = model(features_list, labels, return_embeddings=True)

    print(f"  Logits形状: {outputs['logits'].shape}")
    print(f"  总损失: {outputs['loss'].item():.4f}")
    print(f"  分类损失: {outputs['classification_loss'].item():.4f}")
    print(f"  对比学习损失: {outputs['contrastive_loss'].item():.4f}")
    print(f"  样本嵌入形状: {outputs['embeddings'].shape}")
    print(f"  节点特征形状: {outputs['node_features'].shape}")

    # 前向传播（推理模式）
    print("\n推理模式（无标签）:")
    model.eval()
    with torch.no_grad():
        predictions = model.predict(features_list)

    print(f"  预测结果: {predictions}")
    print(f"  真实标签: {labels}")
    print(f"  准确率: {(predictions == labels).sum().item() / batch_size * 100:.2f}%")

    print("\n" + "=" * 80 + "\n")


def example5_training_step():
    """示例5: 训练步骤示例"""
    print("=" * 80)
    print("示例5: 训练步骤示例")
    print("=" * 80)

    # 参数
    batch_size = 16
    text_dim = 768
    video_dim = 512
    audio_dim = 256
    num_classes = 6

    # 创建模型
    model = MultimodalHypergraphNetwork(
        feature_dims=[text_dim, video_dim, audio_dim],
        hidden_dim=256,
        output_dim=128,
        num_classes=num_classes
    )

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 模拟训练几个步骤
    print("\n开始训练...")
    model.train()

    for step in range(5):
        # 生成模拟数据
        text_features = torch.randn(batch_size, text_dim)
        video_features = torch.randn(batch_size, video_dim)
        audio_features = torch.randn(batch_size, audio_dim)
        labels = torch.randint(0, num_classes, (batch_size,))

        features_list = [text_features, video_features, audio_features]

        # 前向传播
        outputs = model(features_list, labels)
        loss = outputs['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 计算准确率
        _, predicted = torch.max(outputs['logits'], 1)
        accuracy = (predicted == labels).sum().item() / batch_size * 100

        print(f"  Step {step+1}: Loss={loss.item():.4f}, Acc={accuracy:.2f}%")

    print("\n训练完成!")
    print("\n" + "=" * 80 + "\n")


def example6_hypergraph_visualization():
    """示例6: 超图结构可视化信息"""
    print("=" * 80)
    print("示例6: 超图结构分析")
    print("=" * 80)

    # 参数
    num_samples = 5
    num_modalities = 3
    k_neighbors = 2

    # 创建模拟特征
    text_features = torch.randn(num_samples, 128)
    video_features = torch.randn(num_samples, 128)
    audio_features = torch.randn(num_samples, 128)

    features_list = [text_features, video_features, audio_features]

    # 构建超图
    constructor = HypergraphConstructor(num_samples, num_modalities, k_neighbors)
    H = constructor.construct_incidence_matrix(features_list)

    print(f"\n超图统计信息:")
    print(f"  样本数量 N: {num_samples}")
    print(f"  模态数量 M: {num_modalities}")
    print(f"  K最近邻 K: {k_neighbors}")
    print(f"  节点总数: {num_samples * num_modalities}")
    print(f"  超边总数: {num_samples + num_modalities}")

    # 节点度分布
    node_degrees = H.sum(dim=1)
    print(f"\n节点度统计:")
    print(f"  平均节点度: {node_degrees.mean().item():.2f}")
    print(f"  最小节点度: {node_degrees.min().item():.0f}")
    print(f"  最大节点度: {node_degrees.max().item():.0f}")

    # 超边度分布
    hyperedge_degrees = H.sum(dim=0)
    print(f"\n超边度统计:")
    print(f"  样本内超边平均度: {hyperedge_degrees[:num_samples].mean().item():.2f}")
    print(f"  模态间超边平均度: {hyperedge_degrees[num_samples:].mean().item():.2f}")

    # 连接密度
    total_connections = H.sum().item()
    max_connections = num_samples * num_modalities * (num_samples + num_modalities)
    density = total_connections / max_connections * 100

    print(f"\n连接统计:")
    print(f"  总连接数: {int(total_connections)}")
    print(f"  连接密度: {density:.2f}%")

    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("超图多模态融合网络使用示例")
    print("=" * 80 + "\n")

    # 运行所有示例
    example1_hypergraph_construction()
    example2_hypergraph_convolution()
    example3_contrastive_learning()
    example4_full_network()
    example5_training_step()
    example6_hypergraph_visualization()

    print("\n所有示例运行完毕!")
