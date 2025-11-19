import os
import logging
import random
from tqdm import tqdm
from datetime import datetime
from omegaconf import DictConfig, OmegaConf

import json
import importlib
from PIL import Image
import pandas as pd

import numpy as np
import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True 

def seed_init(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def make_saving_folder_and_logger(cfg):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = f"{cfg.general_para.setting}/{timestamp}" # 本次实验文件夹名称
    father_folder_name = cfg.output_para.save_dir # 实验总文件夹名称

    if not os.path.exists(father_folder_name):
        os.makedirs(father_folder_name, exist_ok=True)

    folder_path = os.path.join(father_folder_name, folder_name)
    os.mkdir(folder_path)
    logger = logging.getLogger()
    logger.handlers = []
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler(f'{father_folder_name}/{folder_name}/log.txt')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return father_folder_name, folder_name, logger

def print_init_msg(logger, cfg):
    config_yaml = OmegaConf.to_yaml(cfg)
    logging.info(config_yaml)
    task_set = cfg.general_para.setting
    logger.info(f"{task_set.upper()} Training Starts!")