# 黑盒持续

import os
exc_dir = os.getcwd()
import random

from omegaconf import DictConfig
import pandas as pd
import hydra
import warnings
warnings.filterwarnings("ignore")



from utils import (
    seed_init, )

@hydra.main(config_path="./config", config_name="config_black")
def run_main(cfg: DictConfig):
    os.chdir(exc_dir)# cfg: DictConfig后执行目录会变
    seed_init(cfg.general_para.seed)

   