import json
import os
import shutil

from models import WatermarkModel, WatermarkDetector
from hydra.utils import to_absolute_path
import torch
import numpy as np
import datetime
from rich.progress import track
from torch.utils.data import DataLoader
# from dataset.data import collate_fn, wav_dataset as my_dataset
from dataset.data import collate_fn, oned_dataset as my_dataset

from models import *
import yaml
from libs.modules import *
from libs.modules.seanet import *
from libs.modules.loss import Loss
from libs.modules.segmentation import *
from torch.optim import AdamW
from itertools import chain
import logging
import random

seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

logging_mark = "#" * 20
logging.basicConfig(level=logging.INFO, format="%(message)s")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(configs):
    logging.info('main function')
    process_config, model_config, train_config = configs
    prev_step = 0

    # ------------------- get train dataset
    audios = my_dataset(process_config=process_config, train_config=train_config, flag='train')
    val_audios = my_dataset(process_config=process_config, train_config=train_config, flag='val')

    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size < len(audios)
    # audios_loader = DataLoader(audios, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    # val_audios_loader = DataLoader(val_audios, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    audios_loader = DataLoader(audios, batch_size=batch_size, shuffle=True)
    val_audios_loader = DataLoader(val_audios, batch_size=batch_size, shuffle=False)

    # ------------------- build model
    # channels = model_config["dim"]["channels"]
    # dimension = model_config["dim"]["dimension"]
    # n_filters = model_config["dim"]["filters"]
    # n_residual_layers = model_config["layers"]["n_residual_layers"]
    # ratios = model_config["layers"]["ratios"]
    # activation = model_config["activation"]["act"]
    # activation_params = model_config["activation"]["params"]
    # norm = model_config["normalization"]["norm"]
    # norm_params = model_config["normalization"]["norm_params"]
    # kernel_size = model_config["dim"]["kernel_size"]
    # last_kernel_size = model_config["dim"]["last_kernel_size"]
    # residual_layer_size = model_config["dim"]["residual_layer_size"]
    # dilation_base = model_config["dim"]["dilation_base"]
    # causal = model_config["dim"]["causal"]
    # pad_mode = model_config["dim"]["pad_mode"]
    # true_skip = model_config["dim"]["true_skip"]
    # compress = model_config["dim"]["compress"]
    # lstm = model_config["dim"]["lstm"]
    # disable_norm_outer_blocks = model_config["dim"]["disable_norm_outer_blocks"]
    # trim_right_ratio = model_config["dim"]["trim_right_ratio"]

    encoder = SEANetEncoder()
    decoder = SEANetDecoder()

    generator = WatermarkModel(encoder=encoder, decoder=decoder).to(device)
    detector = WatermarkDetector(dimension=128, output_dim=32, ratios=[1,2]).to(device)

    # ------------------- optimizer
    en_de_op = AdamW(
        params=chain(generator.parameters(), detector.parameters()),
        betas=train_config["optimize"]["betas"],
        eps=train_config["optimize"]["eps"],
        weight_decay=train_config["optimize"]["weight_decay"],
        lr=train_config["optimize"]["lr"]
    )

    # ------------------- loss
    loss = Loss()

    # ------------------- init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    # ------------------- train
    logging.info(logging_mark + "\t" + "Begin Training" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    sample_rate = 16000
    window_size = 1024
    overlap_ratio = 0.2
    bands = [(20, 300), (300, 3000), (3000, 20000)]
    global_step = 0
    train_len = len(audios_loader)
    for ep in range(1, epoch_num + 1):
        generator.train()
        detector.train()
        step = 0
        logging.info("Epoch {}/{}".format(ep, epoch_num))
        for sample in track(audios_loader):
            global_step += 1
            step += 1
            # ------------------- generate watermark
            wav_matrix = sample['matrix'].to(device)
            watermarked_wav, wm = generator(wav_matrix)  # (B, C, L)

            # List of signals from different bands segmentation
            wav_band_signals = divide_signal_into_bands(wav_matrix, sample_rate, bands)
            wm_band_signals = divide_signal_into_bands(wm, sample_rate, bands)


            wav_segments = segment_signal(wav_band_signals, window_size, overlap_ratio)
            wm_segments = segment_signal(wm_band_signals, window_size, overlap_ratio)



            prob, msg = detector.detect_watermark(watermarked_wav)
            losses = loss.en_de_loss(wav_matrix, watermarked_wav, wm, prob)
            if global_step < prev_step:
                sum_loss = lambda_m * losses[1]
            else:
                sum_loss = lambda_e * losses[0] + lambda_m * losses[1]

            en_de_op.zero_grad()
            sum_loss.backward()
            en_de_op.step()

            if step % show_circle == 0:
                logging.info("-" * 100)
                logging.info("training_step:{} - wav_loss:{:.8f} - acc:{:.8f}".format(step, losses[0], losses[1]))

        if ep % save_circle == 0:
            path = os.path.join(train_config["path"]["ckpt"], "pth")
            torch.save({
                "generator": generator.state_dict(),
                "detector": detector.state_dict()
            },
                os.path.join(path, "none-" + generator.name + "_ep_{}_{}.pth.tar".format(ep,
                                                                                         datetime.datetime.now().strftime(
                                                                                             "%Y-%m_%d_%H_%M_%S")))
            )
            shutil.copyfile(os.path.realpath(__file__), os.path.join(path, os.path.basename(os.path.realpath(__file__)))) # save training script

        # ------------------- validation stage
        with torch.no_grad():
            generator.eval()
            detector.eval()
            avg_wav_loss = 0
            avg_acc = 0
            count = 0
            for sample in track(val_audios_loader):
                count += 1
                # ------------------- generate watermark
                wav_matrix = sample["matrix"].to(device)
                watermarked_wav, wm = generator(wav_matrix)
                prob, msg = detector.detect_watermark(watermarked_wav)
                losses = loss.en_de_loss(wav_matrix, watermarked_wav, wm, prob)
                avg_wav_loss += losses[0]
                avg_acc += losses[1]
            avg_wav_loss /= count
            avg_acc /= count
            logging.info("#e" * 60)
            logging.info("eval_epoch:{} - wav_loss:{:.8f} - acc_loss:{:.8f}".format(ep, avg_wav_loss, avg_acc))


if __name__ == "__main__":
    # Read config
    process_config = yaml.load(
        open(r'E:/pythonProject/config/process.yaml', 'r'), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(r'E:/pythonProject/config/model.yaml', 'r'), Loader=yaml.FullLoader)
    train_config = yaml.load(open(r'E:/pythonProject/config/train.yaml', 'r'), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(configs)






