import os
import wandb
import soundfile
import librosa
import matplotlib.pyplot as plt
import matplotlib
import librosa.feature
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

matplotlib.use('Agg')  # Use a non-interactive backend
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
logging_mark = "#" * 20
logging.basicConfig(filename="mylog_{}.log".format(datetime.datetime.now().strftime("%Y-%m_%d_%H_%M_%S")),
                    level=logging.INFO, format="%(message)s")
device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")


def select_random_chunk(audio_data, percentage=0.99):
    batch_size, total_length = audio_data.shape

    # Initialize the label vectors for the entire batch
    label_vector = np.zeros((batch_size, total_length))
    selected_audio_batch = []

    for i in range(batch_size):
        # Randomly select a percentage between 50% and 85%
        # percentage = np.random.uniform(0.5, 0.90)

        # Determine the length of the chunk to be selected
        chunk_length = int(total_length * percentage)

        # Randomly choose the start of the chunk
        start_point = np.random.randint(0, total_length - chunk_length + 1)
        end_point = start_point + chunk_length

        # Create the label vector for this sequence
        label_vector[i, start_point:end_point] = 1

        # Select the audio using the label vector
        selected_audio = audio_data[i][label_vector[i] == 1]
        selected_audio_batch.append(selected_audio)

    return torch.from_numpy(label_vector).float().to(audio_data.device), torch.stack(selected_audio_batch, dim=0)


def save_spectrogram_as_img(audio, datadir, sample_rate=16000, plt_type='mel'):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.png'
    out_path = os.path.join(datadir, timestamp)
    if plt_type == 'spec':
        spec = np.abs(librosa.stft(audio))
        spec_db = librosa.amplitude_to_db(spec, ref=np.max)
    else:
        mel_spec = librosa.feature.melspectrogram(audio, sr=sample_rate)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    fig = plt.Figure()
    ax = fig.add_subplot()
    ax.set_axis_off()

    librosa.display.specshow(
        spec_db if plt_type == 'spec' else mel_spec_db,
        y_axis='log' if plt_type == 'spec' else 'mel',
        x_axis='time', ax=ax)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    fig.savefig(out_path)
    return out_path


def substitute_watermarked_audio(wav_matrix, watermarked_wav, label_vector):
    # Ensure wav_matrix is a torch tensor and on the same device as the watermarked_wav
    wav_matrix = wav_matrix.to(watermarked_wav.device)

    batch_size = wav_matrix.size(0)

    for i in range(batch_size):
        # Get the mask for the current sequence
        mask = label_vector[i].bool()

        # Replace the selected part of wav_matrix with watermarked_wav
        wav_matrix[i][mask] = watermarked_wav[i]

    return wav_matrix


def main(configs):
    logging.info('main function')
    process_config, model_config, train_config = configs
    prev_step = 0
    test_mode = True

    # ------------------- get train dataset
    train_audios = my_dataset(process_config=process_config, train_config=train_config, flag='train')
    val_audios = my_dataset(process_config=process_config, train_config=train_config, flag='val')
    dev_audios = my_dataset(process_config=process_config, train_config=train_config, flag='test')

    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size < len(train_audios)
    train_audios_loader = DataLoader(train_audios, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_audios_loader = DataLoader(val_audios, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    dev_audios_loader = DataLoader(dev_audios, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    wandb.login(key="9a11e5364efe3bb8fedb3741188ee0d714e942e2")
    wandb.init(project="real-time-voice-watermark", name='full_run_20_epoch', config={
        "learning_rate": train_config["optimize"]["lr"],
        "dataset": "LibriSpeech",
        "epochs": train_config["iter"]["epoch"],
    })
    val_audio_table = wandb.Table(columns=['ep', 'Original Audio', "Watermarked Audio", "Watermark Waveform",
                                           "Original Spectrogram", "Watermarked Audio Spectrogram",
                                           "Watermarked Waveform Spectrogram"])
    test_audio_table = wandb.Table(columns=['Original Audio', "Watermarked Audio", "Watermark Waveform",
                                            "Original Spectrogram", "Watermarked_Audio Spectrogram",
                                            "Watermarked Waveform Spectrogram"])
    test_loss_summary_table = wandb.Table(columns=['test_l1_loss', "test_bce_loss", "test_perceptual_loss",
                                                   "test_freq_loss", "test_ber", "test_total_loss"])

    encoder = SimpleEncoder()
    decoder = SimpleDecoder()
    detector = SimpleDetector()

    msgprocessor = MsgProcessor(nbits=train_config["watermark"]["nbits"], hidden_size=1).to(device)

    distortions = distortion()

    wm_generator = WatermarkModel(encoder=encoder, decoder=decoder, msg_processor=msgprocessor,
                                  model_config=model_config).to(device)
    wm_detector = WatermarkDetector(detector=detector, nbits=train_config["watermark"]["nbits"],
                                    model_config=model_config).to(device)

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
    dev_log_path = os.path.join(train_config["path"]["log_path"], "dev")

    val_spec_pth = os.path.join(train_config["path"]['mel_path'], "val_spectrogram")
    test_spec_pth = os.path.join(train_config["path"]['mel_path'], "test_spectrogram")

    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    os.makedirs(dev_log_path, exist_ok=True)
    os.makedirs(val_spec_pth, exist_ok=True)
    os.makedirs(test_spec_pth, exist_ok=True)

    # ------------------- train
    logging.info(logging_mark + "\t" + "Begin Training" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    cnt = 0
    interval = math.ceil(len(train_audios_loader) / 30)
    for ep in range(1, epoch_num + 1):
        wm_generator.train()
        wm_detector.train()
        step = 0
        running_l1_loss = 0.0
        running_binary_cross_entropy_loss = 0.0
        running_perceptual_loss = 0.0
        # running_smoothness_loss = 0.0
        running_freq_loss = 0.0
        running_ber = 0.0
        logging.info("Epoch {}/{}".format(ep, epoch_num))
        for sample in track(train_audios_loader):
            global_step += 1
            step += 1
            en_de_op.zero_grad()
            # ------------------- generate watermark
            orig_wav_matrix = sample['matrix'].to(device)

            physcial_distortions = [4, 5, 6]
            all_distortions = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15]

            # wav_matrix  # Add random distortion, physical distortion type
            if random.random() < 0.5:
                wav_matrix = distortions(orig_wav_matrix, random.choice(physcial_distortions))
            else:
                wav_matrix = orig_wav_matrix

            label_vec, selected_wav_matrix = select_random_chunk(wav_matrix)

            msg = torch.randint(0, 2, (selected_wav_matrix.shape[0], train_config["watermark"]["nbits"]),
                                device=device).float()
            watermarked_wav, wm = wm_generator(selected_wav_matrix, message=msg)  # (B, L)

            # watermarked_wav  # Add random distortion
            # if random.random() < 0.5:
            #     watermarked_wav = distortions(watermarked_wav,  random.choice(all_distortions))

            # Substitute the selected part of wav_matrix with watermarked_wav
            # substituted_wav_matrix has the same shape as wav_matrix
            substituted_wav_matrix = substitute_watermarked_audio(wav_matrix, watermarked_wav, label_vec)

            prob, decoded_msg = wm_detector(substituted_wav_matrix)

            losses = loss.en_de_loss(selected_wav_matrix, watermarked_wav, wm, prob, label_vec, decoded_msg, msg)

            # Convert probabilities to binary values (0 or 1) using a threshold of 0.5
            predicted_bits = (decoded_msg > 0.5).float()

            # Calculate the number of bit errors
            bit_errors = torch.sum(predicted_bits != msg).item()

            # Calculate BER
            total_bits = msg.size(0)  # Total number of bits
            ber = bit_errors / total_bits

            sum_loss = losses[0] + losses[1] + losses[2] + losses[3] + losses[4] + losses[5]

            sum_loss.backward()
            en_de_op.step()

            running_l1_loss += losses[0]
            running_binary_cross_entropy_loss += losses[1]
            running_perceptual_loss += losses[2]
            # running_smoothness_loss += losses[3]
            running_freq_loss += losses[4]
            running_ber += ber

            if step % show_circle == 0:
                logging.info("-" * 100)
                logging.info("training_step:{} - l1_loss:{:.8f} - binary_cross_entropy_loss:{:.8f} - "
                             "perceptual_loss:{:.8f} - freq_loss:{:.8f} - Bit Error Rate:{:.8f}".format(step,
                                                                                                        losses[0],
                                                                                                        losses[1],
                                                                                                        losses[2],
                                                                                                        losses[4],
                                                                                                        ber))

        train_l1_loss = running_l1_loss / len(train_audios_loader)
        train_bce_loss = running_binary_cross_entropy_loss / len(train_audios_loader)
        train_perceptual_loss = running_perceptual_loss / len(train_audios_loader)
        # train_smoothness_loss = running_smoothness_loss / len(train_audios_loader)
        train_freq_loss = running_freq_loss / len(train_audios_loader)
        train_total_loss = (train_l1_loss + train_bce_loss + train_perceptual_loss + train_freq_loss)
        train_ber = running_ber / len(train_audios_loader)
        logging.info("t" * 60)
        logging.info("train_epoch:{} - l1_loss:{:.8f} - bce_loss:{:.8f} - perceptual_loss:{:.8f} - "
                     "train_freq_loss:{:.8f} - train_ber:{:.8f} - total_loss:{:.8f}".
                     format(ep, train_l1_loss, train_bce_loss, train_perceptual_loss, train_freq_loss, train_ber,
                            train_total_loss))

        train_metrics = {"train/train_l1_loss": train_l1_loss,
                         "train/train_bce_loss": train_bce_loss,
                         "train/train_perceptual_loss": train_perceptual_loss,
                         "train/train_freq_loss": train_freq_loss,
                         "train/train_BER": train_ber,
                         "train/total_loss": train_total_loss}

        if ep % save_circle == 0:
            path = train_config["path"]["ckpt"]
            torch.save({
                "generator": wm_generator.state_dict(),
                "detector": wm_detector.state_dict()
            },
                os.path.join(path, "real_time_voice_watermark_ep_{}_{}.pth.tar".
                             format(ep, datetime.datetime.now().strftime("%Y-%m_%d_%H_%M_%S")))
            )
            # shutil.copyfile(os.path.realpath(__file__),
            #                 os.path.join(path, os.path.basename(os.path.realpath(__file__))))  # save training script

        # ------------------- validation stage
        with torch.no_grad():
            wm_generator.eval()
            wm_detector.eval()
            running_l1_loss = 0.0
            running_bce = 0.0
            running_perceptual_loss = 0.0
            # running_smoothness_loss = 0.0
            running_ber = 0.0
            running_freq_loss = 0.0
            for sample in track(val_audios_loader):
                # ------------------- generate watermark
                orig_wav_matrix = sample['matrix'].to(device)

                if True:
                    wav_matrix = orig_wav_matrix

                label_vec, selected_wav_matrix = select_random_chunk(wav_matrix)

                msg = torch.randint(0, 2, (selected_wav_matrix.shape[0], train_config["watermark"]["nbits"]),
                                    device=device).float()

                watermarked_wav, wm = wm_generator(selected_wav_matrix, message=msg)

                # Substitute the selected part of wav_matrix with watermarked_wav
                substituted_wav_matrix = substitute_watermarked_audio(wav_matrix, watermarked_wav, label_vec)

                prob, decoded_msg = wm_detector(substituted_wav_matrix)
                losses = loss.en_de_loss(selected_wav_matrix, watermarked_wav, wm, prob, label_vec, decoded_msg, msg)

                # Convert probabilities to binary values (0 or 1) using a threshold of 0.5
                predicted_bits = (decoded_msg > 0.5).float()

                # Calculate the number of bit errors
                bit_errors = torch.sum(predicted_bits != msg).item()

                # Calculate BER
                total_bits = msg.size(0)  # Total number of bits
                ber = bit_errors / total_bits

                running_l1_loss += losses[0]
                running_bce += losses[1]
                running_perceptual_loss += losses[2]
                # running_smoothness_loss += losses[3]
                running_ber += ber
                running_freq_loss += losses[4]

            val_l1_loss = running_l1_loss / len(val_audios_loader)
            val_bce_loss = running_bce / len(val_audios_loader)
            val_perceptual_loss = running_perceptual_loss / len(val_audios_loader)
            # val_smoothness_loss = running_smoothness_loss / len(val_audios_loader)
            val_freq_loss = running_freq_loss / len(val_audios_loader)
            val_ber = running_ber / len(val_audios_loader)
            val_total_loss = val_l1_loss + val_bce_loss + val_perceptual_loss + val_freq_loss

            val_metrics = {"val/val_l1_loss": val_l1_loss,
                           "val/val_bce_loss": val_bce_loss,
                           "val/val_perceptual_loss": val_perceptual_loss,
                           "val/val_freq_loss": val_freq_loss,
                           "val/val_BER": val_ber,
                           "val/val_total_loss": val_total_loss}

            if ep % interval == 0:
                # Compute the spectrogram
                spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=320, hop_length=160)

                original_spectrogram = spectrogram_transform(orig_wav_matrix[-1].cpu())
                watermarked_audio_spectrogram = spectrogram_transform(substituted_wav_matrix[-1].cpu())
                watermark_wav_spectrogram = spectrogram_transform(wm[-1].cpu())

                # Convert the spectrogram to a format suitable for matplotlib
                # Convert the spectrogram to dB scale for better visualization
                original_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(original_spectrogram)
                watermarked_audio_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(watermarked_audio_spectrogram)
                watermark_wm_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(watermark_wav_spectrogram)

                # Set the common x-axis limit based on the original spectrogram
                x_min, x_max = 0, original_spectrogram_db.shape[-1]  # The number of time frames (x-axis)

                # Get the min and max values from the original spectrogram for consistent scaling
                vmin = original_spectrogram_db.min().item()
                vmax = original_spectrogram_db.max().item()

                # Plot the original spectrogram
                plt.figure(figsize=(10, 4))
                plt.imshow(original_spectrogram_db.numpy(), cmap='viridis', origin='lower', aspect='auto', vmin=vmin,
                           vmax=vmax)
                plt.xlim(x_min, x_max)  # Set x-axis limits
                plt.colorbar(format="%+2.0f dB")
                plt.title("Original Spectrogram")
                plt.xlabel("Time")
                plt.ylabel("Frequency")

                # Save the original spectrogram image
                original_spectrogram_path = os.path.join(val_spec_pth, "{}_epoch_{}_original_spectrogram.png".
                                                         format(datetime.datetime.now().
                                                                strftime("%Y-%m_%d_%H_%M_%S"), ep))
                plt.savefig(original_spectrogram_path)
                plt.close()

                # Plot the watermarked audio spectrogram
                plt.figure(figsize=(10, 4))
                plt.imshow(watermarked_audio_spectrogram_db.numpy(), cmap='viridis', origin='lower', aspect='auto',
                           vmin=vmin, vmax=vmax)
                plt.xlim(x_min, x_max)  # Set x-axis limits
                plt.colorbar(format="%+2.0f dB")
                plt.title("Watermarked Audio Spectrogram")
                plt.xlabel("Time")
                plt.ylabel("Frequency")

                # Save the watermarked spectrogram image
                watermarked_spectrogram_path = os.path.join(val_spec_pth,
                                                            "{}_epoch_{}_watermarked_audio_spectrogram.png".format(
                                                                datetime.datetime.now().strftime(
                                                                    "%Y-%m_%d_%H_%M_%S"), ep))
                plt.savefig(watermarked_spectrogram_path)
                plt.close()

                # Plot the watermark wm spectrogram
                plt.figure(figsize=(10, 4))
                plt.imshow(watermark_wm_spectrogram_db.numpy(), cmap='viridis', origin='lower', aspect='auto',
                           vmin=vmin, vmax=vmax)
                plt.xlim(x_min, x_max)  # Set x-axis limits
                plt.colorbar(format="%+2.0f dB")
                plt.title("Watermark Spectrogram")
                plt.xlabel("Time")
                plt.ylabel("Frequency")

                # Save the watermark wm spectrogram image
                watermark_wm_spectrogram_path = os.path.join(val_spec_pth, "{}_epoch_{}_watermark_spectrogram.png".
                                                             format(datetime.datetime.now().
                                                                    strftime("%Y-%m_%d_%H_%M_%S"), ep))
                plt.savefig(watermark_wm_spectrogram_path)
                plt.close()

                val_audio_table.add_data(ep,
                                         wandb.Audio(orig_wav_matrix[-1].cpu().numpy(), sample_rate=16000),
                                         wandb.Audio(watermarked_wav[-1].cpu().numpy(), sample_rate=16000),
                                         wandb.Audio(wm[-1].cpu().numpy(), sample_rate=16000),
                                         wandb.Image(original_spectrogram_path),
                                         wandb.Image(watermarked_spectrogram_path),
                                         wandb.Image(watermark_wm_spectrogram_path))

            wandb.log({**train_metrics, **val_metrics})
            logging.info("#e" * 60)
            logging.info("eval_epoch:{} - l1_loss:{:.8f} - bce_loss:{:.8f} - perceptual_loss:{:.8f} - "
                         "freq_loss:{:.8f} - BER:{:.8f} - total_loss:{:.8f}".format(ep,
                                                                                    val_l1_loss,
                                                                                    val_bce_loss,
                                                                                    val_perceptual_loss,
                                                                                    val_freq_loss,
                                                                                    val_ber,
                                                                                    val_total_loss))
    wandb.log({'val_audio_table': val_audio_table})

    # ------------------- test stage
    with torch.no_grad():
        wm_generator.eval()
        wm_detector.eval()
        running_l1_loss = 0.0
        running_bce_loss = 0.0
        running_perceptual_loss = 0.0
        # running_smoothness_loss = 0.0
        running_ber = 0.0
        running_freq_loss = 0.0
        steps = 0
        interval = math.ceil(len(dev_audios_loader) / 5)
        for sample in track(dev_audios_loader):
            steps += 1
            orig_wav_matrix = sample["matrix"].to(device)

            if True:
                wav_matrix = orig_wav_matrix

            label_vec, selected_wav_matrix = select_random_chunk(wav_matrix)

            msg = torch.randint(0, 2, (selected_wav_matrix.shape[0], train_config["watermark"]["nbits"]),
                                device=device).float()

            watermarked_wav, wm = wm_generator(selected_wav_matrix, message=msg)

            # Substitute the selected part of wav_matrix with watermarked_wav
            substituted_wav_matrix = substitute_watermarked_audio(wav_matrix, watermarked_wav, label_vec)

            prob, decoded_msg = wm_detector(substituted_wav_matrix)
            losses = loss.en_de_loss(selected_wav_matrix, watermarked_wav, wm, prob, label_vec, decoded_msg, msg)

            # Convert probabilities to binary values (0 or 1) using a threshold of 0.5
            predicted_bits = (decoded_msg > 0.5).float()

            # Calculate the number of bit errors
            bit_errors = torch.sum(predicted_bits != msg).item()

            # Calculate BER
            total_bits = msg.size(0)  # Total number of bits
            ber = bit_errors / total_bits

            running_l1_loss += losses[0]
            running_bce_loss += losses[1]
            running_perceptual_loss += losses[2]
            # running_smoothness_loss += losses[3]
            running_ber += ber
            running_freq_loss += losses[4]

            if steps % interval == 0:
                # soundfile.write(os.path.join("./results/wm_speech", "selected_original_{}.wav".format(steps)),
                #                 selected_wav_matrix[0].cpu().squeeze(0).detach().numpy(),
                #                 samplerate=16000, format="WAV")
                # soundfile.write(os.path.join("./results/wm_speech", "watermark_{}.wav".format(steps)),
                #                 wm[0].cpu().squeeze(0).detach().numpy(), samplerate=16000, format="WAV")
                # soundfile.write(os.path.join("./results/wm_speech", "watermarked_{}.wav".format(steps)),
                #                 watermarked_wav[0].cpu().squeeze(0).detach().numpy(), samplerate=16000, format="WAV")

                # Compute the spectrogram
                spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=320, hop_length=160)

                original_spectrogram = spectrogram_transform(orig_wav_matrix[0].cpu())
                watermarked_audio_spectrogram = spectrogram_transform(substituted_wav_matrix[0].cpu())
                watermark_wav_spectrogram = spectrogram_transform(wm[0].cpu())

                # Convert the spectrogram to a format suitable for matplotlib
                # Convert the spectrogram to dB scale for better visualization
                original_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(original_spectrogram)
                watermarked_audio_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(
                    watermarked_audio_spectrogram)
                watermark_wm_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(watermark_wav_spectrogram)

                # Set the common x-axis limit based on the original spectrogram
                x_min, x_max = 0, original_spectrogram_db.shape[-1]  # The number of time frames (x-axis)

                # Get the min and max values from the original spectrogram for consistent scaling
                vmin = original_spectrogram_db.min().item()
                vmax = original_spectrogram_db.max().item()

                # Plot the original spectrogram
                plt.figure(figsize=(10, 4))
                plt.imshow(original_spectrogram_db.numpy(), cmap='viridis', origin='lower', aspect='auto', vmin=vmin,
                           vmax=vmax)
                plt.xlim(x_min, x_max)  # Set x-axis limits
                plt.colorbar(format="%+2.0f dB")
                plt.title("Original Spectrogram")
                plt.xlabel("Time")
                plt.ylabel("Frequency")

                # Save the original audio spectrogram image
                original_spectrogram_path = os.path.join(test_spec_pth, "{}_original_spectrogram.png".
                                                         format(datetime.datetime.now().
                                                                strftime("%Y-%m_%d_%H_%M_%S")))
                plt.savefig(original_spectrogram_path)
                plt.close()

                # Plot the watermarked audio spectrogram
                plt.figure(figsize=(10, 4))
                plt.imshow(watermarked_audio_spectrogram_db.numpy(), cmap='viridis', origin='lower', aspect='auto',
                           vmin=vmin, vmax=vmax)
                plt.xlim(x_min, x_max)  # Set x-axis limits
                plt.colorbar(format="%+2.0f dB")
                plt.title("Watermarked Audio Spectrogram")
                plt.xlabel("Time")
                plt.ylabel("Frequency")

                # Save the watermarked audio spectrogram image
                watermarked_spectrogram_path = os.path.join(test_spec_pth, "{}_watermarked_audio_spectrogram.png"
                                                            .format(datetime.datetime.now().
                                                                    strftime("%Y-%m_%d_%H_%M_%S")))
                plt.savefig(watermarked_spectrogram_path)
                plt.close()

                # Plot the watermark wm spectrogram
                plt.figure(figsize=(10, 4))
                plt.imshow(watermark_wm_spectrogram_db.numpy(), cmap='viridis', origin='lower', aspect='auto',
                           vmin=vmin, vmax=vmax)
                plt.xlim(x_min, x_max)  # Set x-axis limits
                plt.colorbar(format="%+2.0f dB")
                plt.title("Watermark Spectrogram")
                plt.xlabel("Time")
                plt.ylabel("Frequency")

                # Save the watermark wm spectrogram image
                watermark_wm_spectrogram_path = os.path.join(test_spec_pth,
                                                             "{}_watermark_spectrogram.png".
                                                             format(datetime.datetime.now().
                                                                    strftime("%Y-%m_%d_%H_%M_%S")))
                plt.savefig(watermark_wm_spectrogram_path)
                plt.close()

                test_audio_table.add_data(wandb.Audio(orig_wav_matrix[-1].cpu().numpy(), sample_rate=16000),
                                          wandb.Audio(watermarked_wav[-1].cpu().numpy(), sample_rate=16000),
                                          wandb.Audio(wm[-1].cpu().numpy(), sample_rate=16000,),
                                          wandb.Image(original_spectrogram_path),
                                          wandb.Image(watermarked_spectrogram_path),
                                          wandb.Image(watermark_wm_spectrogram_path))

        test_l1_loss = running_l1_loss / len(dev_audios_loader)
        test_bce_loss = running_bce / len(dev_audios_loader)
        test_perceptual_loss = running_perceptual_loss / len(dev_audios_loader)
        test_freq_loss = running_freq_loss / len(dev_audios_loader)
        test_ber = running_ber / len(dev_audios_loader)
        test_total_loss = test_l1_loss + test_bce_loss + test_perceptual_loss + test_freq_loss

        test_loss_summary_table.add_data(test_l1_loss, test_bce_loss, test_perceptual_loss, test_freq_loss,
                                         test_ber, test_total_loss)

        wandb.log({"test_audio_table": test_audio_table, "test_loss_summary_table": test_loss_summary_table})
        logging.info("#test#" * 20)
        logging.info("test l1_loss:{:.8f} - BCE_loss:{:.8f} - perceptual_loss:{:.8f} - "
                     "freq_loss:{:.8f} - BER:{:.8f} - total_loss:{:.8f}".format(test_l1_loss,
                                                                                test_bce_loss,
                                                                                test_perceptual_loss,
                                                                                test_freq_loss,
                                                                                test_ber,
                                                                                test_total_loss))
    wandb.finish()


if __name__ == "__main__":
    # Read config
    process_config = yaml.load(
        open(r'./config/process.yaml', 'r'), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(r'./config/model.yaml', 'r'), Loader=yaml.FullLoader)
    train_config = yaml.load(open(r'./config/train.yaml', 'r'), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)
    main(configs)
    torch.cuda.empty_cache()
    # try:
    #     main(configs)
    # except Exception as e:
    #     error_type = type(e).__name__  # 获取异常类型的名称
    #     logging.error(f"An error of type {error_type} occurred: {e}")
