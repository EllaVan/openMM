"""
超图网络使用示例

演示如何使用超图网络进行多模态情感分类
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hypergraph_network import HypergraphEmotionClassifier
from emotion_dataloader import create_dataloaders


def example_1_basic_usage():
    """
    示例 1: 基础使用
    """
    print("\n" + "="*60)
    print("示例 1: 基础使用")
    print("="*60)

    # 创建模拟数据
    batch_size = 8
    T = 50  # 时间步数

    batch = {
        'text_features': torch.randn(batch_size, T, 768),
        'audio_features': torch.randn(batch_size, T, 768),
        'video_features': torch.randn(batch_size, T, 768),
        'label': torch.randint(0, 7, (batch_size,))
    }

    # 创建模型
    feature_dims = {'text': 768, 'audio': 768, 'video': 768}

    model = HypergraphEmotionClassifier(
        feature_dims=feature_dims,
        num_classes=7
    )

    # 前向传播
    output = model(batch)

    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"Classification Loss: {output['cls_loss'].item():.4f}")
    if 'contrastive_loss' in output:
        print(f"Contrastive Loss: {output['contrastive_loss'].item():.4f}")

    # 预测
    predictions = model.predict(batch)
    print(f"Predictions: {predictions}")


def example_2_custom_config():
    """
    示例 2: 自定义配置
    """
    print("\n" + "="*60)
    print("示例 2: 自定义配置")
    print("="*60)

    # 自定义配置
    config = {
        'encoder_hidden_dim': 256,
        'encoder_output_dim': 512,
        'hypergraph_hidden_dim': 512,
        'num_hyperedges': 128,
        'num_conv_layers': 3,
        'bottleneck_dim': 256,
        'dropout': 0.2,
        'hyperedge_drop_rate': 0.3,
        'use_contrastive': True,
        'contrastive_weight': 0.2,
        'use_bottleneck': True
    }

    feature_dims = {'text': 768, 'audio': 768, 'video': 768}

    model = HypergraphEmotionClassifier(
        feature_dims=feature_dims,
        num_classes=7,
        config=config
    )

    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")


def example_3_with_dataloader():
    """
    示例 3: 使用 DataLoader
    """
    print("\n" + "="*60)
    print("示例 3: 使用 DataLoader")
    print("="*60)

    # 检查数据是否存在
    data_dir = './output/mosei'
    if not os.path.exists(data_dir):
        print("⚠ 数据目录不存在，请先运行特征提取")
        print(f"  python extract_dataset_features.py")
        return

    # 加载数据
    dataloaders = create_dataloaders(
        data_dir=data_dir,
        dataset_name='MOSEI',
        emotion='happy',
        label_id=0,
        batch_size=16,
        num_workers=2
    )

    train_loader = dataloaders['train']

    # 获取一个 batch
    batch = next(iter(train_loader))

    print(f"Batch keys: {batch.keys()}")
    print(f"Text features: {batch['text_features'].shape}")
    print(f"Audio features: {batch['audio_features'].shape}")
    print(f"Video features: {batch['video_features'].shape}")
    print(f"Labels: {batch['label'].shape}")

    # 创建模型
    feature_dims = {
        'text': batch['text_features'].shape[-1],
        'audio': batch['audio_features'].shape[-1],
        'video': batch['video_features'].shape[-1]
    }

    model = HypergraphEmotionClassifier(
        feature_dims=feature_dims,
        num_classes=2
    )

    # 前向传播
    output = model(batch)

    print(f"\nOutput keys: {output.keys()}")
    print(f"Logits: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")


def example_4_training_loop():
    """
    示例 4: 训练循环
    """
    print("\n" + "="*60)
    print("示例 4: 训练循环")
    print("="*60)

    # 创建模拟数据
    batch_size = 16
    T = 50

    # 创建模型
    feature_dims = {'text': 768, 'audio': 768, 'video': 768}
    model = HypergraphEmotionClassifier(
        feature_dims=feature_dims,
        num_classes=7
    )

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 训练几个 step
    model.train()
    for step in range(5):
        # 模拟数据
        batch = {
            'text_features': torch.randn(batch_size, T, 768),
            'audio_features': torch.randn(batch_size, T, 768),
            'video_features': torch.randn(batch_size, T, 768),
            'label': torch.randint(0, 7, (batch_size,))
        }

        # 前向传播
        output = model(batch)
        loss = output['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step+1}: Loss = {loss.item():.4f}")


def example_5_hypergraph_visualization():
    """
    示例 5: 超图连接矩阵可视化
    """
    print("\n" + "="*60)
    print("示例 5: 超图连接矩阵分析")
    print("="*60)

    # 创建模拟数据
    batch_size = 4
    T = 20

    batch = {
        'text_features': torch.randn(batch_size, T, 768),
        'audio_features': torch.randn(batch_size, T, 768),
        'video_features': torch.randn(batch_size, T, 768),
        'label': torch.randint(0, 7, (batch_size,))
    }

    # 创建模型
    feature_dims = {'text': 768, 'audio': 768, 'video': 768}
    config = {'num_hyperedges': 32}

    model = HypergraphEmotionClassifier(
        feature_dims=feature_dims,
        num_classes=7,
        config=config
    )

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(batch)

    H = output['H']  # [batch, 3T, num_hyperedges]

    print(f"超图连接矩阵 H:")
    print(f"  形状: {H.shape}")
    print(f"  节点数: {H.shape[1]} (3T = 3 × {T})")
    print(f"  超边数: {H.shape[2]}")
    print(f"  数值范围: [{H.min():.4f}, {H.max():.4f}]")

    # 分析第一个样本
    H_0 = H[0]  # [3T, num_hyperedges]

    # 每个节点连接的超边数量
    node_degrees = (H_0 > 0.01).sum(dim=1)
    print(f"\n节点度分析:")
    print(f"  平均度数: {node_degrees.float().mean():.2f}")
    print(f"  最大度数: {node_degrees.max()}")
    print(f"  最小度数: {node_degrees.min()}")

    # 每个超边包含的节点数量
    edge_degrees = (H_0 > 0.01).sum(dim=0)
    print(f"\n超边度分析:")
    print(f"  平均大小: {edge_degrees.float().mean():.2f}")
    print(f"  最大超边: {edge_degrees.max()} 个节点")
    print(f"  最小超边: {edge_degrees.min()} 个节点")

    # 分析跨模态连接
    text_nodes = H_0[:T, :]  # 文本节点
    audio_nodes = H_0[T:2*T, :]  # 音频节点
    video_nodes = H_0[2*T:3*T, :]  # 视频节点

    print(f"\n跨模态连接分析:")
    print(f"  文本-音频共享超边: {((text_nodes > 0.01) & (audio_nodes > 0.01)).sum()}")
    print(f"  文本-视频共享超边: {((text_nodes > 0.01) & (video_nodes > 0.01)).sum()}")
    print(f"  音频-视频共享超边: {((audio_nodes > 0.01) & (video_nodes > 0.01)).sum()}")
    print(f"  三模态共享超边: {((text_nodes > 0.01) & (audio_nodes > 0.01) & (video_nodes > 0.01)).sum()}")


def main():
    print("="*60)
    print("超图网络使用示例")
    print("="*60)
    print("\n可用示例:")
    print("  1. 基础使用")
    print("  2. 自定义配置")
    print("  3. 使用 DataLoader")
    print("  4. 训练循环")
    print("  5. 超图连接矩阵分析")
    print("  0. 运行所有示例")

    choice = input("\n请选择示例 (0-5): ").strip()

    if choice == '1':
        example_1_basic_usage()
    elif choice == '2':
        example_2_custom_config()
    elif choice == '3':
        example_3_with_dataloader()
    elif choice == '4':
        example_4_training_loop()
    elif choice == '5':
        example_5_hypergraph_visualization()
    elif choice == '0':
        example_1_basic_usage()
        example_2_custom_config()
        example_3_with_dataloader()
        example_4_training_loop()
        example_5_hypergraph_visualization()
    else:
        print("无效选择")

    print("\n" + "="*60)
    print("示例运行完成")
    print("="*60)


if __name__ == "__main__":
    main()
