"""
Zero-shot Continual Learning 主训练脚本

整合所有组件：
1. DataLoader (seen/unseen分离)
2. AUEmotionNetwork (backbone + AU分支)
3. TwoStageTrainer (阶段1: seen训练, 阶段2: unseen zero-shot EM迭代)
4. Beta先验管理
"""

import os
import sys
import yaml
import logging
import torch
from pathlib import Path
from datetime import datetime

# 添加项目路径
code_root = Path(__file__).parent
project_root = code_root.parent
sys_path = [str(project_root), str(code_root)]
sys.path.extend(sys_path)

from core import (
    AUEmotionNetwork,
    load_au_emo_prior
)
from data.dataloader import create_task_dataloaders_separated, IncrementalLabelMapper
from training.two_stage_trainer import TwoStageTrainer


def setup_logger(log_dir: str) -> logging.Logger:
    """设置日志记录器"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}.log"

    # 创建logger
    logger = logging.getLogger() #logger = logging.getLogger('ZeroshotCL')
    logger.setLevel(logging.INFO)

    # 文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 格式化器
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"日志文件: {log_file}")

    return logger


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model(config: dict, logger: logging.Logger) -> AUEmotionNetwork:
    """创建模型"""
    logger.info("创建模型...")

    # 加载AU-EMO先验
    au_emo_prior = None
    if config['paths']['au_emo_prior']:
        au_emo_prior, au_names, emotion_names = load_au_emo_prior(config['paths']['au_emo_prior'])
        logger.info(f"加载AU-EMO先验: {config['paths']['au_emo_prior']}")
        logger.info(f"先验形状: {au_emo_prior.shape}")

    # 创建模型
    model = AUEmotionNetwork(
        text_input_dim=config['model']['text_input_dim'],
        audio_input_dim=config['model']['audio_input_dim'],
        video_input_dim=config['model']['video_input_dim'],
        num_aus=len(au_names), #config['model']['num_aus'],
        num_emotions=len(emotion_names), #config['model']['num_emotions'],
        encoder_hidden_dim=config['model']['encoder_hidden_dim'],
        encoder_output_dim=config['model']['encoder_output_dim'],
        hypergraph_hidden_dim=config['model']['hypergraph_hidden_dim'],
        num_hyperedges=config['model']['num_hyperedges'],
        num_conv_layers=config['model']['num_conv_layers'],
        dropout=config['model']['dropout'],
        au_emo_prior=au_emo_prior,
        prior_strength=config['model']['prior_strength'],
        device=config['device']
    )

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"\n模型创建完成:")
    logger.info(f"总参数量: {total_params:,}")
    logger.info(f"可训练参数: {trainable_params:,}")

    return model


def main():
    """主函数"""
    # 1. 加载配置
    config_path = "config/train_config.yaml"
    config = load_config(config_path)

    # 2. 设置日志
    logger = setup_logger(config['output']['log_dir'])
    
    logger.info("# Zero-shot Continual Learning Training")
    
    logger.info(f"配置文件: {config_path}")
    logger.info(f"设备: {config['device']}")

    # 3. 设置设备
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")

    if device.type == 'cuda':
        logger.info(f"  GPU名称: {torch.cuda.get_device_name(0)}")
        logger.info(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 4. 创建模型
    model = create_model(config, logger)

    # 5. 创建训练器
    logger.info("创建训练器...")

    # 合并配置（为了兼容trainer）
    trainer_config = {
        'model': config['model'],
        'training': config['training'],
        'continual_learning': config['continual_learning'],
        'output': config['output'],
        'pseudo_count': config['beta_prior']['pseudo_count'],
        'au_emo_prior_path': config['paths']['au_emo_prior'],
        'au_embedding_path': config['paths']['au_embedding'],
        'zeroshot_lr': config['training']['zeroshot_lr'],
        'zeroshot_hidden_layers': config['training']['zeroshot_hidden_layers'],
        'convergence_threshold': config['training']['convergence_threshold']
    }

    trainer = TwoStageTrainer(
        model=model,
        config=trainer_config,
        logger=logger,
        device=str(device)
    )

    logger.info("训练器创建完成")

    # 6. 加载任务数据
    logger.info("加载任务数据...")

    task_config_path = config['data']['task_config_path']
    label_mapper = IncrementalLabelMapper()

    # 读取任务配置
    import json
    with open(task_config_path, 'r') as f:
        task_config = json.load(f)

    num_tasks = len(task_config['tasks'])
    logger.info(f"总任务数: {num_tasks}")

    # 7. 逐任务训练
    for task_id in range(num_tasks):
        logger.info(f"# 任务 {task_id + 1}/{num_tasks}")

        # 加载任务数据
        train_loaders, test_loaders, label_mapper, task_info = create_task_dataloaders_separated(
            task_config_path=task_config_path,
            task_id=task_id,
            label_mapper=label_mapper,
            batch_size=config['data']['batch_size'],
            num_workers=config['data']['num_workers'],
            train_ratio=config['data']['train_ratio'],
            shuffle_train=config['data']['shuffle_train'],
            seed=config['data']['seed']
        )

        # 训练任务
        task_stats = trainer.train_task(
            task_id=task_id,
            task_name=task_info['task_name'],
            task_info=task_info,
            train_loaders=train_loaders,
            test_loaders=test_loaders,
            num_epochs_stage1=config['training']['stage1_epochs'],
            num_em_iterations=config['training']['em_iterations'],
            num_epochs_per_em=config['training']['epochs_per_em']
        )

        # 打印任务总结
        logger.info(f"任务 {task_id} 训练完成")

    # 8. 训练完成
    logger.info("# 全部训练完成！")

    logger.info(f"\n总类别数: {label_mapper.get_num_classes_so_far()}")
    logger.info(f"标签映射: {label_mapper.original_to_incremental}")

    # 保存最终模型
    final_save_path = Path(config['output']['save_dir']) / 'final_model.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_mapper': label_mapper,
        'config': config
    }, final_save_path)

    logger.info(f"\n最终模型已保存: {final_save_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n训练出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
