"""
训练基于超图的多模态融合网络

使用情感数据集（MOSEI或MELD）训练模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse
import os

from hypergraph_model import MultimodalHypergraphNetwork
from emotion_dataloader import create_dataloaders


def collate_fn(batch):
    """
    自定义collate函数，提取多模态特征

    假设batch中每个样本是字典:
    {
        'audio_features': tensor (seq_len, feature_dim),
        'text_features': tensor (seq_len, feature_dim),
        'video_features': tensor (seq_len, feature_dim),
        'label': int
    }
    """
    audio_features = []
    text_features = []
    video_features = []
    labels = []

    for item in batch:
        # 对序列特征取平均池化，得到固定维度的特征向量
        audio_feat = torch.tensor(item['audio_features']).float()
        text_feat = torch.tensor(item['text_features']).float()
        video_feat = torch.tensor(item['video_features']).float()

        # 如果是3维，需要先去掉batch维度
        if audio_feat.dim() == 3:
            audio_feat = audio_feat.squeeze(0)
        if text_feat.dim() == 3:
            text_feat = text_feat.squeeze(0)
        if video_feat.dim() == 3:
            video_feat = video_feat.squeeze(0)

        # 平均池化
        audio_feat = audio_feat.mean(dim=0)  # (feature_dim,)
        text_feat = text_feat.mean(dim=0)
        video_feat = video_feat.mean(dim=0)

        audio_features.append(audio_feat)
        text_features.append(text_feat)
        video_features.append(video_feat)
        labels.append(item['label'])

    # 堆叠成batch
    audio_features = torch.stack(audio_features)  # (batch_size, audio_dim)
    text_features = torch.stack(text_features)    # (batch_size, text_dim)
    video_features = torch.stack(video_features)  # (batch_size, video_dim)
    labels = torch.tensor(labels)                  # (batch_size,)

    return {
        'audio': audio_features,
        'text': text_features,
        'video': video_features,
        'labels': labels
    }


def train_epoch(model, train_loader, optimizer, device):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_cls_loss = 0
    total_con_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for batch in pbar:
        # 移动数据到设备
        audio = batch['audio'].to(device)
        text = batch['text'].to(device)
        video = batch['video'].to(device)
        labels = batch['labels'].to(device)

        # 前向传播
        features_list = [text, video, audio]  # 按text/video/audio顺序
        outputs = model(features_list, labels)

        loss = outputs['loss']
        cls_loss = outputs['classification_loss']
        con_loss = outputs['contrastive_loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_con_loss += con_loss.item()

        _, predicted = torch.max(outputs['logits'], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100 * correct / total:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    avg_cls_loss = total_cls_loss / len(train_loader)
    avg_con_loss = total_con_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, avg_cls_loss, avg_con_loss, accuracy


def evaluate(model, test_loader, device):
    """评估模型"""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for batch in pbar:
            # 移动数据到设备
            audio = batch['audio'].to(device)
            text = batch['text'].to(device)
            video = batch['video'].to(device)
            labels = batch['labels'].to(device)

            # 前向传播
            features_list = [text, video, audio]
            outputs = model(features_list, labels)

            loss = outputs['loss']
            total_loss += loss.item()

            # 预测
            _, predicted = torch.max(outputs['logits'], 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # 更新进度条
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })

    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy, all_predictions, all_labels


def main(args):
    """主训练函数"""
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据加载器
    print(f"\nLoading {args.dataset} dataset...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        emotion=args.emotion,
        label_id=args.label_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed,
        collate_fn=collate_fn
    )

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # 获取特征维度
    sample_batch = next(iter(train_loader))
    text_dim = sample_batch['text'].shape[1]
    video_dim = sample_batch['video'].shape[1]
    audio_dim = sample_batch['audio'].shape[1]

    print(f"\nFeature dimensions:")
    print(f"  Text: {text_dim}")
    print(f"  Video: {video_dim}")
    print(f"  Audio: {audio_dim}")

    # 创建模型
    model = MultimodalHypergraphNetwork(
        feature_dims=[text_dim, video_dim, audio_dim],
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_classes=args.num_classes,
        num_hgcn_layers=args.num_hgcn_layers,
        k_neighbors=args.k_neighbors,
        dropout=args.dropout,
        temperature=args.temperature
    ).to(device)

    print(f"\nModel created with {sum(p.numel() for p in model.parameters())} parameters")

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    # 训练
    best_acc = 0
    best_epoch = 0

    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 80)

    for epoch in range(args.epochs):
        print(f"\nEpoch [{epoch+1}/{args.epochs}]")

        # 训练
        train_loss, train_cls_loss, train_con_loss, train_acc = train_epoch(
            model, train_loader, optimizer, device
        )

        # 评估
        test_loss, test_acc, _, _ = evaluate(model, test_loader, device)

        # 学习率调度
        scheduler.step()

        # 打印结果
        print(f"\nTrain Loss: {train_loss:.4f} (Cls: {train_cls_loss:.4f}, Con: {train_con_loss:.4f})")
        print(f"Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc: {test_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch + 1

            if args.save_model:
                os.makedirs('checkpoints', exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'test_acc': test_acc,
                }, f'checkpoints/best_model_{args.dataset}_{args.emotion}.pt')
                print(f"Saved best model with acc: {test_acc:.2f}%")

    print("\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best Test Accuracy: {best_acc:.2f}% at epoch {best_epoch}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hypergraph Multimodal Network')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='Data',
                        help='数据目录')
    parser.add_argument('--dataset', type=str, default='MOSEI',
                        choices=['MOSEI', 'MELD'],
                        help='数据集名称')
    parser.add_argument('--emotion', type=str, default='happy',
                        help='情感类型')
    parser.add_argument('--label_id', type=int, default=0,
                        help='标签ID')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例（仅MOSEI）')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 模型参数
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='隐藏层维度')
    parser.add_argument('--output_dim', type=int, default=128,
                        help='输出嵌入维度')
    parser.add_argument('--num_classes', type=int, default=6,
                        help='分类类别数')
    parser.add_argument('--num_hgcn_layers', type=int, default=2,
                        help='超图卷积层数量')
    parser.add_argument('--k_neighbors', type=int, default=5,
                        help='K最近邻的K值')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout概率')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='对比学习温度参数')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--lr_step', type=int, default=20,
                        help='学习率衰减步长')
    parser.add_argument('--lr_gamma', type=float, default=0.5,
                        help='学习率衰减系数')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载进程数')

    # 其他参数
    parser.add_argument('--save_model', action='store_true',
                        help='是否保存模型')

    args = parser.parse_args()

    main(args)
