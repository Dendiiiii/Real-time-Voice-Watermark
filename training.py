import json
import os
import shutil
import wandb
import librosa
import matplotlib.pyplot as plt

from models import WatermarkModel, WatermarkDetector
import torch
import numpy as np
import datetime
from rich.progress import track
from torch.utils.data import DataLoader
from dataset.data import collate_fn, wav_dataset as my_dataset

from Distortions.distortions import *
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
logging.basicConfig(filename="mylog_{}.log".format(datetime.datetime.now().strftime("%Y-%m_%d_%H_%M_%S")),
                    level=logging.INFO, format="%(message)s")
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")


def save_spectrogram_as_img(audio, datadir, sample_rate=16000, plt_type='mel'):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    out_path = os.path.join(datadir, timestamp)
    print(type(audio))
    if plt_type == 'spec':
        spec = np.abs(librosa.stft(audio))
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    else:
        print('test_save_func_1')
        mel_spec = librosa.feature.melspectrogram(audio, sr=sample_rate)
        print('test_save_func_2')
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        print('test_save_func_3')
    fig = plt.Figure()
    print('test_save_func_4')
    ax = fig.add_subplot()
    print('test_save_func_5')
    ax.set_axis_off()
    print('test_save_func_6')

    librosa.display.specshow(
        spec_db if plt_type == 'spec' else mel_spec_db,
        y_axis='log' if plt_type == 'spec' else 'mel',
        x_axis='time', ax=ax)
    print('test_save_func_7')

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    fig.savefig(out_path)
    print('test_save_func_8')
    return out_path


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
    wandb.login(key="9a11e5364efe3bb8fedb3741188ee0d714e942e2")
    wandb.init(project="real-time-voice-watermark", name='full_run_20_epoch', config={
        "learning_rate": train_config["optimize"]["lr"],
        "dataset": "LibriSpeech",
        "epochs": train_config["iter"]["epoch"],
    })
    audio_table = wandb.Table(columns=['Original Audio', 'Watermarked Audio'])
    encoder = SimpleEncoder()
    decoder = SimpleDecoder()
    detector = SimpleDetector()
    distortions = distortion()

    wm_generator = WatermarkModel(encoder=encoder, decoder=decoder).to(device)
    wm_detector = WatermarkDetector(detector=detector, nbits=0).to(device)

    # ------------------- optimizer
    en_de_op = AdamW(
        params=chain(wm_generator.parameters(), wm_detector.parameters()),
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
    # mel_spectrogram_path = train_config["path"]["mel_path"]
    # wm_mel_spectrogram_path = os.path.join(train_config["path"]["mel_path"], "wm")

    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    # os.makedirs(mel_spectrogram_path, exist_ok=True)
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
        wm_generator.train()
        wm_detector.train()
        step = 0
        running_l1_loss = 0.0
        running_binary_cross_entropy_loss = 0.0
        running_perceptual_loss = 0.0
        logging.info("Epoch {}/{}".format(ep, epoch_num))
        for sample in track(audios_loader):
            global_step += 1
            step += 1
            en_de_op.zero_grad()
            # ------------------- generate watermark
            wav_matrix = sample['matrix'].to(device)
            physcial_distortions = [4, 5, 6]
            all_distortions = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15]

            # wav_matrix  # Add random distortion, physical distortion type
            if random.random() < 0.5:
                wav_matrix = distortions(wav_matrix, random.choice(physcial_distortions))

            watermarked_wav, wm = wm_generator(wav_matrix)  # (B, L)

            # watermarked_wav  # Add random distortion
            # if random.random() < 0.5:
            #     watermarked_wav = distortions(watermarked_wav,  random.choice(all_distortions))

            masks = (torch.rand(watermarked_wav.size()[0]) < 0.5).to(device)
            detect_data = wav_matrix.clone()
            reshaped_masks = masks.unsqueeze(1).expand_as(detect_data)
            detect_data[reshaped_masks] = watermarked_wav[reshaped_masks]

            prob, msg = wm_detector(detect_data)

            losses = loss.en_de_loss(wav_matrix, watermarked_wav, wm, prob, reshaped_masks)

            sum_loss = losses[0] + losses[1] + 1e6 * losses[2]

            sum_loss.backward()
            en_de_op.step()

            running_l1_loss += losses[0]
            running_binary_cross_entropy_loss += losses[1]
            running_perceptual_loss += losses[2]

            if step % show_circle == 0:
                logging.info("-" * 100)
                logging.info("training_step:{} - l1_loss:{:.8f} - binary_cross_entropy_loss:{:.8f} - "
                             "perceptual_loss:{:.8f}".format(step, losses[0], losses[1], losses[2]))

        train_l1_loss = running_l1_loss / len(audios_loader)
        train_binary_cross_entropy_loss = running_binary_cross_entropy_loss / len(audios_loader)
        train_perceptual_loss = running_perceptual_loss / len(audios_loader)
        total_loss = train_l1_loss + train_binary_cross_entropy_loss + train_perceptual_loss
        logging.info("t" * 60)
        logging.info("train_epoch:{} - l1_loss:{:.8f} - binary_cross_entropy_loss:{:.8f} - perceptual_loss:{:.8f}".
                     format(ep, train_l1_loss, train_binary_cross_entropy_loss, train_perceptual_loss))

        train_metrics = {"train/train_l1_loss": train_l1_loss,
                         "train/train_binary_cross_entropy_loss": train_binary_cross_entropy_loss,
                         "train/train_perceptual_loss": train_perceptual_loss,
                         "train/total_loss": total_loss}

        if ep % save_circle == 0:
            path = train_config["path"]["ckpt"]
            torch.save({
                "generator": wm_generator.state_dict(),
                "detector": wm_detector.state_dict()
            },
                os.path.join(path, "real_time_voice_watermark_ep_{}_{}.pth.tar".
                             format(ep, datetime.datetime.now().strftime("%Y-%m_%d_%H_%M_%S")))
            )
            shutil.copyfile(os.path.realpath(__file__),
                            os.path.join(path, os.path.basename(os.path.realpath(__file__)))) # save training script

        # ------------------- validation stage
        with torch.no_grad():
            wm_generator.eval()
            wm_detector.eval()
            running_l1_loss = 0.0
            running_bce = 0.0
            running_perceptual_loss = 0.0
            for sample in track(val_audios_loader):
                # ------------------- generate watermark
                wav_matrix = sample["matrix"].to(device)
                watermarked_wav, wm = wm_generator(wav_matrix)
                masks = (torch.rand(watermarked_wav.size()[0]) < 0.5).to(device)
                detect_data = wav_matrix.clone()
                reshaped_masks = masks.unsqueeze(1).expand_as(detect_data)
                detect_data[reshaped_masks] = watermarked_wav[reshaped_masks]
                prob, msg = wm_detector(detect_data)
                losses = loss.en_de_loss(wav_matrix, watermarked_wav, wm, prob, reshaped_masks)
                running_l1_loss += losses[0]
                running_bce += losses[1]
                running_perceptual_loss += losses[2]

            val_l1_loss = running_l1_loss / len(val_audios_loader)
            val_bce = running_bce / len(val_audios_loader)
            val_perceptual_loss = running_perceptual_loss / len(val_audios_loader)
            val_total_loss = val_l1_loss + val_bce + 1e6 * val_perceptual_loss

            val_metrics = {"val/val_l1_loss": val_l1_loss,
                           "val/val_binary_cross_entropy_loss": val_bce,
                           "val/val_perceptual_loss": val_perceptual_loss,
                           "val/val_total_loss": val_total_loss}
            print("1")
            spec_pth = os.path.join(train_config["path"]['mel_path'], 'audio_melspec')
            melspec_pth = os.path.join(train_config["path"]['mel_path'], 'wm_melspec')
            print("2")
            # wav_mel_pth = save_spectrogram_as_img(wav_matrix[-1].cpu().numpy(), spec_pth)
            # print("2.1")
            # wm_mel_pth = save_spectrogram_as_img(watermarked_wav[-1].cpu().numpy(), melspec_pth)
            # print("3")
            audio_table.add_data(wandb.Audio(wav_matrix[-1].cpu().numpy()),
                                 wandb.Audio(watermarked_wav[-1].cpu().numpy()))
            print("4")
            wandb.log({**train_metrics, **val_metrics})
            logging.info("#e" * 60)
            logging.info("eval_epoch:{} - l1_loss:{:.8f} - binary_cross_entropy_loss:{:.8f} - perceptual_loss:{:.8f}".
                         format(ep, val_l1_loss, val_bce, val_perceptual_loss))
    wandb.finish()


if __name__ == "__main__":
    # Read config
    process_config = yaml.load(
        open(r'./config/process.yaml', 'r'), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(r'./config/model.yaml', 'r'), Loader=yaml.FullLoader)
    train_config = yaml.load(open(r'./config/train.yaml', 'r'), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)
    try:
        main(configs)
    except Exception as e:
        error_type = type(e).__name__  # 获取异常类型的名称
        logging.error(f"An error of type {error_type} occurred: {e}")






