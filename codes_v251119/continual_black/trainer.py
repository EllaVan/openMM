import os
from omegaconf import DictConfig

from utils.core_tools import make_saving_folder_and_logger
from learnable_matrix import LearnableAUEMOMatrix, load_au_emo_prior
from dataloader_continual import create_emotion_dataloaders

from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, cfg: DictConfig, task_id: str):
        self.cfg = cfg
        self.task_id = task_id
        
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

        self.dataloaders = create_emotion_dataloaders(self.logger, self.cfg)