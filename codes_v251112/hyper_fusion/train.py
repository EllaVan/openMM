#!/usr/bin/env python
"""
超图融合网络训练脚本 - 支持 Padding + Masking
"""

import os
import sys
import argparse
import logging
import torch
import torch.optim as optim
from datetime import datetime
from pathlib import Path

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from hyper_fusion.dataloader import create_dataloaders
from hyper_fusion.network import HypergraphEmotionClassifier


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> dict:
    """
    训练一个 epoch

    Args:
        model: 模型
        dataloader: 数据加载器
        optimizer: 优化器
        device: 设备
        epoch: 当前 epoch

    Returns:
        metrics: 训练指标
    """
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_contrastive_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        # 移动数据到设备
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()}

        # 前向传播
        output = model(batch)
        loss = output['loss']

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_cls_loss += output['cls_loss'].item()
        total_contrastive_loss += output['contrastive_loss'].item()

        predictions = torch.argmax(output['logits'], dim=1)
        correct += (predictions == batch['labels']).sum().item()
        total += batch['labels'].size(0)

        # 打印进度
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {loss.item():.4f} | "
                f"Acc: {100.*correct/total:.2f}%"
            )

    # 计算平均指标
    metrics = {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'contrastive_loss': total_contrastive_loss / len(dataloader),
        'accuracy': correct / total
    }

    return metrics


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    """
    评估模型

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 设备

    Returns:
        metrics: 评估指标
    """
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            # 移动数据到设备
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # 前向传播
            output = model(batch)

            # 统计
            total_loss += output['loss'].item()
            total_cls_loss += output['cls_loss'].item()

            predictions = torch.argmax(output['logits'], dim=1)
            correct += (predictions == batch['labels']).sum().item()
            total += batch['labels'].size(0)

    # 计算平均指标
    metrics = {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'accuracy': correct / total
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='训练超图融合网络')

    # 数据参数
    parser.add_argument('--data_dir', type=str, required=True,
                       help='数据目录')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['MOSEI', 'MELD'],
                       help='数据集名称')
    parser.add_argument('--emotion', type=str, required=True,
                       help='情感类型 (e.g., happy, sad)')
    parser.add_argument('--label_id', type=int, required=True,
                       help='标签 ID')

    # 模型参数
    parser.add_argument('--encoder_hidden_dim', type=int, default=256,
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
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='训练集比例 (MOSEI)')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')

    # 其他参数
    parser.add_argument('--use_contrastive', action='store_true', default=True,
                       help='使用对比学习')
    parser.add_argument('--contrastive_weight', type=float, default=0.1,
                       help='对比学习损失权重')
    parser.add_argument('--use_bottleneck', action='store_true', default=True,
                       help='使用 Bottleneck 层')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载进程数')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                       help='模型保存目录')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='设备')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 加载数据
    logger.info("加载数据...")
    dataloaders = create_dataloaders(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        emotion=args.emotion,
        label_id=args.label_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=args.train_ratio,
        seed=args.seed,
        shuffle_train=True,
        pin_memory=(device.type == 'cuda')
    )

    train_loader = dataloaders['train']
    test_loader = dataloaders['test']

    # 获取特征维度
    batch = next(iter(train_loader))
    feature_dims = {
        'text': batch['text_features'].shape[-1],
        'audio': batch['audio_features'].shape[-1],
        'video': batch['video_features'].shape[-1]
    }

    logger.info(f"特征维度: {feature_dims}")

    # 创建模型
    logger.info("创建模型...")
    config = {
        'encoder_hidden_dim': args.encoder_hidden_dim,
        'encoder_output_dim': args.encoder_output_dim,
        'hypergraph_hidden_dim': args.hypergraph_hidden_dim,
        'num_hyperedges': args.num_hyperedges,
        'num_conv_layers': args.num_conv_layers,
        'bottleneck_dim': args.bottleneck_dim,
        'dropout': args.dropout,
        'hyperedge_drop_rate': args.hyperedge_drop_rate,
        'use_contrastive': args.use_contrastive,
        'use_bottleneck': args.use_bottleneck,
        'contrastive_weight': args.contrastive_weight
    }

    model = HypergraphEmotionClassifier(
        feature_dims=feature_dims,
        num_classes=2,
        config=config
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数数量: {num_params:,}")

    # 创建优化器
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
    logger.info("开始训练...")
    logger.info(f"{'='*60}")

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch
        )

        # 评估
        test_metrics = evaluate(model, test_loader, device)

        # 打印结果
        logger.info(f"{'='*60}")
        logger.info(f"Epoch {epoch}/{args.epochs}")
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Cls Loss: {train_metrics['cls_loss']:.4f}, "
            f"Contrastive Loss: {train_metrics['contrastive_loss']:.4f}, "
            f"Acc: {100.*train_metrics['accuracy']:.2f}%"
        )
        logger.info(
            f"Test  - Loss: {test_metrics['loss']:.4f}, "
            f"Cls Loss: {test_metrics['cls_loss']:.4f}, "
            f"Acc: {100.*test_metrics['accuracy']:.2f}%"
        )

        # 学习率调度
        scheduler.step(test_metrics['accuracy'])

        # 保存最佳模型
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            best_epoch = epoch

            save_path = os.path.join(
                args.save_dir,
                f"best_model_{args.dataset}_{args.emotion}.pth"
            )

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': best_acc,
                'config': config,
                'feature_dims': feature_dims,
                'args': vars(args)
            }, save_path)

            logger.info(f"✓ 保存最佳模型 (Acc: {100.*best_acc:.2f}%)")

        logger.info(f"{'='*60}\n")

    # 训练完成
    logger.info(f"{'='*60}")
    logger.info(f"训练完成！")
    logger.info(f"最佳测试准确率: {100.*best_acc:.2f}% (Epoch {best_epoch})")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
