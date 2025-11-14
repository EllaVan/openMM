"""
超图网络训练脚本

使用提取的 MOSEI/MELD 特征训练超图情感分类模型
"""

import os
import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging

from hypergraph_network import HypergraphEmotionClassifier
from emotion_dataloader import create_dataloaders


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """训练一个 epoch"""
    model.train()

    total_loss = 0
    total_cls_loss = 0
    total_contrastive_loss = 0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')

    for batch in pbar:
        # 移动到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # 前向传播
        output = model(batch)

        loss = output['loss']
        logits = output['logits']
        labels = batch['label']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_cls_loss += output.get('cls_loss', 0).item()
        if 'contrastive_loss' in output:
            total_contrastive_loss += output['contrastive_loss'].item()

        predictions = torch.argmax(logits, dim=1)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{total_correct/total_samples:.4f}'
        })

    # 计算平均
    metrics = {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'contrastive_loss': total_contrastive_loss / len(dataloader),
        'accuracy': total_correct / total_samples
    }

    return metrics


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> dict:
    """评估模型"""
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # 移动到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 前向传播
            output = model(batch)

            if 'loss' in output:
                total_loss += output['loss'].item()

            logits = output['logits']
            labels = batch['label']

            predictions = torch.argmax(logits, dim=1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 计算指标
    metrics = {
        'loss': total_loss / len(dataloader) if total_loss > 0 else 0,
        'accuracy': total_correct / total_samples,
        'predictions': np.array(all_predictions),
        'labels': np.array(all_labels)
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='训练超图情感分类模型')

    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--dataset', type=str, default='MOSEI',
                       choices=['MOSEI', 'MELD'],
                       help='数据集名称')
    parser.add_argument('--emotion', type=str, default='happy',
                       help='情感类型')
    parser.add_argument('--label_id', type=int, default=0,
                       help='标签 ID')

    # 模型参数
    parser.add_argument('--encoder_hidden_dim', type=int, default=128,
                       help='编码器隐藏层维度')
    parser.add_argument('--encoder_output_dim', type=int, default=256,
                       help='编码器输出维度')
    parser.add_argument('--hypergraph_hidden_dim', type=int, default=256,
                       help='超图隐藏层维度')
    parser.add_argument('--num_hyperedges', type=int, default=64,
                       help='超边数量')
    parser.add_argument('--num_conv_layers', type=int, default=2,
                       help='超图卷积层数')
    parser.add_argument('--bottleneck_dim', type=int, default=128,
                       help='Bottleneck 维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout 率')
    parser.add_argument('--hyperedge_drop_rate', type=float, default=0.2,
                       help='超边删除率')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=50,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='权重衰减')
    parser.add_argument('--use_contrastive', action='store_true',
                       help='使用对比学习')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                       help='对比学习权重')
    parser.add_argument('--use_bottleneck', action='store_true',
                       help='使用 Bottleneck 层')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例 (仅 MOSEI)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'使用设备: {device}')

    # 加载数据
    logger.info('加载数据...')
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        emotion=args.emotion,
        label_id=args.label_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio
    )

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    logger.info(f'训练集大小: {len(train_loader.dataset)}')
    logger.info(f'测试集大小: {len(test_loader.dataset)}')

    # 获取特征维度
    sample_batch = next(iter(train_loader))
    feature_dims = {
        'text': sample_batch['text_features'].shape[-1],
        'audio': sample_batch['audio_features'].shape[-1],
        'video': sample_batch['video_features'].shape[-1]
    }

    logger.info(f'特征维度: {feature_dims}')

    # 创建模型
    logger.info('创建模型...')
    model_config = {
        'encoder_hidden_dim': args.encoder_hidden_dim,
        'encoder_output_dim': args.encoder_output_dim,
        'hypergraph_hidden_dim': args.hypergraph_hidden_dim,
        'num_hyperedges': args.num_hyperedges,
        'num_conv_layers': args.num_conv_layers,
        'bottleneck_dim': args.bottleneck_dim,
        'dropout': args.dropout,
        'hyperedge_drop_rate': args.hyperedge_drop_rate,
        'use_contrastive': args.use_contrastive,
        'contrastive_weight': args.contrastive_weight,
        'use_bottleneck': args.use_bottleneck
    }

    model = HypergraphEmotionClassifier(
        feature_dims=feature_dims,
        num_classes=2,  # 二分类 (该情感 vs 其他)
        config=model_config
    ).to(device)

    logger.info(f'模型参数数量: {sum(p.numel() for p in model.parameters())}')

    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # 训练
    logger.info('开始训练...')
    best_accuracy = 0

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch
        )

        logger.info(
            f'Epoch {epoch} [Train] - '
            f'Loss: {train_metrics["loss"]:.4f}, '
            f'Cls Loss: {train_metrics["cls_loss"]:.4f}, '
            f'Contrastive Loss: {train_metrics["contrastive_loss"]:.4f}, '
            f'Acc: {train_metrics["accuracy"]:.4f}'
        )

        # 评估
        test_metrics = evaluate(model, test_loader, device)

        logger.info(
            f'Epoch {epoch} [Test] - '
            f'Loss: {test_metrics["loss"]:.4f}, '
            f'Acc: {test_metrics["accuracy"]:.4f}'
        )

        # 学习率调度
        scheduler.step(test_metrics['accuracy'])

        # 保存最佳模型
        if test_metrics['accuracy'] > best_accuracy:
            best_accuracy = test_metrics['accuracy']

            save_path = os.path.join(
                args.save_dir,
                f'best_model_{args.dataset}_{args.emotion}.pth'
            )

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_accuracy,
                'config': model_config,
                'feature_dims': feature_dims
            }, save_path)

            logger.info(f'保存最佳模型: {save_path} (Acc: {best_accuracy:.4f})')

    logger.info(f'训练完成！最佳准确率: {best_accuracy:.4f}')


if __name__ == '__main__':
    main()
