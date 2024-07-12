import json
import os
import shutil
import wandb
from torch.utils.tensorboard import SummaryWriter

from models import WatermarkModel, WatermarkDetector
from hydra.utils import to_absolute_path
import torch
import numpy as np
import datetime
from rich.progress import track
from torch.utils.data import DataLoader
from dataset.data import collate_fn, wav_dataset as my_dataset

from Distortions import *
from models import *
import yaml
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
    audios_loader = DataLoader(audios, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_audios_loader = DataLoader(val_audios, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    wandb.require("core")
    wandb.init(project="real-time-voice-watermark", name='experiment_2', config={
        "learning_rate": train_config["optimize"]["lr"],
        "dataset": "LibriSpeech",
        "epochs": train_config["iter"]["epoch"],
    })

    encoder = SimpleEncoder()
    decoder = SimpleDecoder()
    detector = SimpleDetector()

    generator = WatermarkModel(encoder=encoder, decoder=decoder).to(device)
    detector = WatermarkDetector(detector=detector, nbits=0).to(device)

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
    # wandb_path = os.path.join(train_config["path"]["log_path"], "learning_rate_"+train_config["optimize"]["lr"])

    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    # os.makedirs(wandb_path, exist_ok=True)

    # ------------------- train
    logging.info(logging_mark + "\t" + "Begin Training" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    for ep in range(1, epoch_num + 1):
        generator.train()
        detector.train()
        step = 0
        running_loudness_loss = 0.0
        running_binary_cross_entropy_loss = 0.0
        logging.info("Epoch {}/{}".format(ep, epoch_num))
        for sample in track(audios_loader):
            global_step += 1
            step += 1
            en_de_op.zero_grad()
            # ------------------- generate watermark
            wav_matrix = sample['matrix'].to(device)

            # wav_matrix  # Add random distortion, physical distortion type

            watermarked_wav, wm = generator(wav_matrix)  # (B, L)

            # watermarked_wav  # Add random distortion

            masks = (torch.rand(watermarked_wav.size()[0]) < 0.5).to(device)
            detect_data = wav_matrix.clone()
            reshaped_masks = masks.unsqueeze(1).expand_as(detect_data)
            detect_data[reshaped_masks] = watermarked_wav[reshaped_masks]

            prob, msg = detector(detect_data)

            losses = loss.en_de_loss(wav_matrix, watermarked_wav, wm, prob, reshaped_masks)

            metrics = {"train/train_loudness_loss": losses[0],
                       "train/train_binary_cross_entropy_loss": losses[1]}
            wandb.log(metrics)

            sum_loss = losses[0] + losses[1]

            sum_loss.backward()
            en_de_op.step()

            running_loudness_loss += losses[0]
            running_binary_cross_entropy_loss += losses[1]

            if step % show_circle == 0:
                logging.info("-" * 100)
                logging.info("training_step:{} - loudness_loss:{:.8f} - binary_cross_entropy_loss:{:.8f}".format(step, losses[0], losses[1]))

        train_loudness_loss = running_loudness_loss / len(audios_loader)
        train_binary_cross_entropy_loss = running_binary_cross_entropy_loss / len(audios_loader)

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

                masks = (torch.rand(watermarked_wav.size()[0]) < 0.5).to(device)
                detect_data = wav_matrix.clone()
                reshaped_masks = masks.view(-1, 1, 1).expand_as(detect_data)
                detect_data[reshaped_masks] = watermarked_wav[reshaped_masks]
                prob, msg = detector(detect_data)
                losses = loss.en_de_loss(wav_matrix, watermarked_wav, wm, prob, reshaped_masks)
                avg_wav_loss += losses[0]
                avg_acc += losses[1]
            avg_wav_loss /= count
            avg_acc /= count
            val_metrics = {"val/val_loudness_loss": avg_wav_loss,
                           "val/val_binary_cross_entropy_loss": avg_acc}
            wandb.log(val_metrics)
            logging.info("#e" * 60)
            logging.info("eval_epoch:{} - loudness_loss:{:.8f} - binary_cross_entropy_loss:{:.8f}".format(ep, avg_wav_loss, avg_acc))
        wandb.finish()


if __name__ == "__main__":
    # Read config
    process_config = yaml.load(
        open(r'D:/Real-time vocie watermark/config/process.yaml', 'r'), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(r'D:/Real-time vocie watermark/config/model.yaml', 'r'), Loader=yaml.FullLoader)
    train_config = yaml.load(open(r'D:/Real-time vocie watermark/config/train.yaml', 'r'), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(configs)






