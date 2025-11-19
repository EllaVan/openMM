"""
跨域零样本持续学习 - 主入口

训练流程：
1. 加载配置文件
2. 初始化模型和组件
3. 顺序训练3个任务
4. 评估和保存结果
"""

import os
import sys
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim

# 添加当前目录到路径
sys.path.append(str(Path(__file__).parent))

from core import (
    AUEmotionNetwork,
    LearnableAUEMOMatrix,
    load_au_emo_prior
)
from data import create_task_dataloaders, IncrementalLabelMapper
from training import ContinualLearningTrainer
from utils import seed_init, setup_logger, count_parameters


@hydra.main(version_base=None, config_path="config", config_name="continual_learning")
def main(cfg: DictConfig):
    """
    主训练函数

    使用Hydra加载配置文件
    """

    # ========================================================================
    # 1. 初始化
    # ========================================================================
    print("\n" + "="*80)
    print("跨域零样本持续学习 - 开始训练")
    print("="*80)

    # 设置随机种子
    seed_init(cfg.system.seed)

    # 设备
    device = cfg.system.device
    print(f"\n使用设备: {device}")

    # 创建logger
    log_file = os.path.join(cfg.output.log_dir, 'training.log')
    logger = setup_logger('ContinualLearning', log_file)

    logger.info("="*80)
    logger.info("配置信息:")
    logger.info("="*80)
    logger.info(OmegaConf.to_yaml(cfg))

    # ========================================================================
    # 2. 加载AU-EMO先验
    # ========================================================================
    logger.info("\n" + "-"*80)
    logger.info("加载AU-EMO先验...")
    logger.info("-"*80)

    prior_matrix, au_names, emotion_names = load_au_emo_prior(cfg.prior.au_prior_path)
    num_emotions = len(emotion_names)

    logger.info(f"先验矩阵形状: {prior_matrix.shape}")
    logger.info(f"AU数量: {len(au_names)}")
    logger.info(f"情绪类别: {emotion_names}")

    # ========================================================================
    # 3. 创建增量标签映射器
    # ========================================================================
    label_mapper = IncrementalLabelMapper()

    # ========================================================================
    # 4. 创建网络模型（集成AU-EMO矩阵）
    # ========================================================================
    logger.info("\n" + "-"*80)
    logger.info("创建AU情绪识别网络（集成AU-EMO矩阵）...")
    logger.info("-"*80)

    model = AUEmotionNetwork(
        text_input_dim=cfg.network.text_dim,
        audio_input_dim=cfg.network.audio_dim,
        video_input_dim=cfg.network.video_dim,
        num_aus=cfg.prior.num_aus,
        num_emotions=num_emotions,
        encoder_hidden_dim=cfg.network.encoder_hidden_dim,
        encoder_output_dim=cfg.network.encoder_output_dim,
        hypergraph_hidden_dim=cfg.network.hypergraph_hidden_dim,
        num_hyperedges=cfg.network.num_hyperedges,
        num_conv_layers=cfg.network.num_conv_layers,
        dropout=cfg.network.dropout,
        au_emo_prior=prior_matrix,
        prior_strength=cfg.prior.prior_strength,
        device=device
    )

    logger.info(f"网络初始化完成（含AU-EMO矩阵）")
    logger.info(f"  先验强度: {cfg.prior.prior_strength}")

    initial_stats = model.au_emo_matrix.get_statistics()
    logger.info(f"  初始KL散度: {initial_stats['kl_from_prior']:.6f}")

    # 统计参数量
    param_stats = count_parameters(model)
    logger.info(f"模型参数统计:")
    logger.info(f"  总参数: {param_stats['total']:,}")
    logger.info(f"  可训练参数: {param_stats['trainable']:,}")

    # ========================================================================
    # 5. 创建优化器
    # ========================================================================
    logger.info("\n" + "-"*80)
    logger.info("创建优化器...")
    logger.info("-"*80)

    # 分别设置网络和矩阵的学习率
    # 收集所有非矩阵参数
    network_params = []
    for name, param in model.named_parameters():
        if 'au_emo_matrix.matrix_logits' not in name:
            network_params.append(param)

    optimizer = optim.Adam([
        {
            'params': network_params,
            'lr': cfg.training.learning_rate
        },
        {
            'params': [model.au_emo_matrix.matrix_logits],
            'lr': cfg.training.matrix_learning_rate
        }
    ], weight_decay=cfg.training.weight_decay)

    logger.info(f"优化器: Adam")
    logger.info(f"  网络学习率: {cfg.training.learning_rate}")
    logger.info(f"  矩阵学习率: {cfg.training.matrix_learning_rate}")
    logger.info(f"  权重衰减: {cfg.training.weight_decay}")

    # ========================================================================
    # 6. 创建训练器
    # ========================================================================
    logger.info("\n" + "-"*80)
    logger.info("创建持续学习训练器...")
    logger.info("-"*80)

    # 将配置转换为字典
    config_dict = OmegaConf.to_container(cfg, resolve=True)

    trainer = ContinualLearningTrainer(
        model=model,
        optimizer=optimizer,
        config=config_dict,
        logger=logger,
        device=device
    )

    logger.info(f"训练器初始化完成")
    logger.info(f"  EWC: {cfg.continual_learning.use_ewc}")
    logger.info(f"  一致性策略: {cfg.continual_learning.consistency_strategy}")

    # ========================================================================
    # 8. 持续学习训练循环
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("开始持续学习训练")
    logger.info("="*80)

    # 读取任务数量
    import json
    with open(cfg.task.config_path, 'r') as f:
        task_config = json.load(f)

    num_tasks = task_config['num_tasks']
    logger.info(f"总任务数: {num_tasks}")

    # 逐任务训练
    for task_id in range(num_tasks):
        logger.info(f"\n{'#'*80}")
        logger.info(f"# 加载 Task {task_id}")
        logger.info(f"{'#'*80}")

        # 加载任务数据
        train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
            task_config_path=cfg.task.config_path,
            task_id=task_id,
            label_mapper=label_mapper,
            batch_size=cfg.dataloader.batch_size,
            num_workers=cfg.dataloader.num_workers,
            train_ratio=cfg.dataloader.train_ratio,
            shuffle_train=cfg.dataloader.shuffle_train,
            seed=cfg.system.seed
        )

        # 训练任务
        task_stats = trainer.train_task(
            task_id=task_id,
            task_name=task_info['task_name'],
            task_info=task_info,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=cfg.training.epochs_per_task
        )

        logger.info(f"\nTask {task_id} 训练完成!")
        logger.info(f"  最终测试准确率: {task_stats['epochs'][-1]['test_acc']:.4f}")

        # 打印当前标签映射
        logger.info(f"\n当前全局标签映射:")
        logger.info(f"  {label_mapper.original_to_incremental}")
        logger.info(f"  总类数: {label_mapper.get_num_classes_so_far()}")

    # ========================================================================
    # 9. 保存最终结果
    # ========================================================================
    logger.info("\n" + "="*80)
    logger.info("保存最终结果")
    logger.info("="*80)

    trainer.save_final_model()

    # 打印最终统计
    logger.info("\n" + "="*80)
    logger.info("训练完成总结")
    logger.info("="*80)

    logger.info(f"\n总任务数: {num_tasks}")
    logger.info(f"总类数: {label_mapper.get_num_classes_so_far()}")

    logger.info(f"\n全局标签映射:")
    for original, incremental in sorted(label_mapper.original_to_incremental.items()):
        logger.info(f"  原始标签 {original} -> 增量标签 {incremental}")

    final_matrix_stats = model.au_emo_matrix.get_statistics()
    logger.info(f"\n最终AU-EMO矩阵统计:")
    for key, value in final_matrix_stats.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("\n" + "="*80)
    logger.info("🎉 跨域零样本持续学习训练完成！")
    logger.info("="*80)

    print("\n✓ 训练完成！查看日志: " + log_file)


if __name__ == "__main__":
    main()
