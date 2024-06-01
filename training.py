import json
from models import WatermarkModel, WatermarkDetector
from hydra.utils import to_absolute_path
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.data import wav_dataset as my_dataset
from models import *
import logging
import random


seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
logging_mark = "#"*20




def main(args, configs):
    logging.info('main function')
    process_config, train_config = configs

    audios = my_dataset(process_config=process_config, train_config=train_config, flag='train')
    val_audios = my_dataset(process_config=process_config, train_config=train_config, flag='val')

    audios_loader = DataLoader(audios, batch_size=64, shuffle=True)
    val_audios_loader = DataLoader(val_audios, batch_size=64, shuffle=False)

    # -------------------train
    logging.info(logging_mark + "\t" + "Begin Training" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    train_len = len(audios_loader)
    # for ep in rang(1, epoch_num+1):






