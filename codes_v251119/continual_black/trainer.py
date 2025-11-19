import os
from omegaconf import DictConfig

from utils.core_tools import make_saving_folder_and_logger
from learnable_matrix import LearnableAUEMOMatrix, load_au_emo_prior
from dataloader_continual import create_task_dataloaders, IncrementalLabelMapper

from utils import (
    DomainSplitter, create_predefined_task_sequence
)

from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        
        self.epochs_per_task = cfg.general_para.epochs_per_task
        self.total_epochs = cfg.general_para.total_epochs

        self.dataset = cfg.data_para.dataset
        self.task = cfg.data_para.dataset

        # set log
        self.father_folder_name, self.folder_name, self.logger = make_saving_folder_and_logger(cfg)
        self.writer = SummaryWriter(os.path.join(self.father_folder_name, self.folder_name))

        prior_matrix, au_names, emotion_names = load_au_emo_prior(cfg.priori_para.au_prior_path)
        num_emotions = len(emotion_names)
        self.logger.info(f"Prior matrix shape: {prior_matrix.shape}")
        self.logger.info(f"AUs: {len(au_names)}")
        self.logger.info(f"Emotions: {emotion_names}")

        # if cfg.task_para.task_config_path:
        #     tasks = DomainSplitter.load_task_configs(cfg.task_para.task_config_path)
        # self.logger.info(f"Number of tasks: {len(tasks)}")
        # for task in tasks:
        #     self.logger.info(f"  {task}")

        # 创建全局标签映射器
        label_mapper = IncrementalLabelMapper()
        # 存储所有任务的信息
        all_task_info = []
        # 顺序加载4个任务
        print("\n[Loadding Task]")
        for task_id in range(3):
            print(f"\n{'='*80}")
            print(f"加载 Task {task_id}")
            print(f"{'='*80}")
            try:
                train_loader, test_loader, label_mapper, task_info = create_task_dataloaders(
                    task_config_path=cfg.task_para.task_config_path,
                    task_id=task_id,
                    label_mapper=label_mapper,
                    batch_size=cfg.dataloader.batch_size,
                    num_workers=cfg.dataloader.num_workers,
                    train_ratio=cfg.dataloader.train_ratio
                )

                all_task_info.append(task_info)

                # 显示任务详情
                print(f"\n任务详情:")
                print(f"  数据集: {task_info['dataset_name']}")
                print(f"  数据目录: {task_info['data_dir']}")
                print(f"  Seen情绪: {task_info['seen_emotions']}")
                print(f"  Unseen情绪: {task_info['unseen_emotions']}")
                print(f"  训练样本: {task_info['train_stats']['total']}")
                print(f"  测试样本: {task_info['test_stats']['total']}")
                print(f"  当前总类数: {task_info['num_classes_so_far']}")

                # 查看一个batch
                if len(train_loader) > 0:
                    batch = next(iter(train_loader))
                    print(f"\n  Batch示例:")
                    print(f"    Shape: text={batch['text'].shape}, audio={batch['audio'].shape}, video={batch['video'].shape}")
                    print(f"    Labels: {batch['label'].tolist()[:5]}... (前5个)")
                    print(f"    Is seen: {batch['is_seen'].tolist()[:5]}... (前5个)")

            except FileNotFoundError as e:
                print(f"\n  ⚠️  警告: 数据文件未找到")
                print(f"     {e}")
                print(f"     跳过此任务，继续下一个...")
                continue