import os
import random

import torch
import torchaudio
from torch.utils.data import Dataset


def collate_fn(batch):
    # Find the maximum length in the batch for padding
    max_len = max([sample["matrix"].size(1) for sample in batch])

    # Pad each sample and update pad_num accordingly
    for sample in batch:
        original_len = sample["matrix"].size(1)
        pad_size = max_len - original_len
        sample["matrix"] = torch.nn.functional.pad(
            sample["matrix"], (0, pad_size)
        )  # Pad with zeros
        sample["pad_num"] = pad_size  # Update the padding number

    # Combine all samples into a single batch dictionary
    batched_sample = {
        "matrix": torch.cat([sample["matrix"] for sample in batch], dim=0),
        "sample_rate": torch.tensor([sample["sample_rate"] for sample in batch]),
        "name": [sample["name"] for sample in batch],
    }
    return batched_sample


# class wav_dataset(Dataset):
#     def __init__(self, process_config, train_config, flag="train"):
#         self.dataset_name = train_config["dataset"]
#         raw_dataset_path = train_config["path"]["raw_path"]
#         self.dataset_path = os.path.join(raw_dataset_path, flag)
#         self.sample_rate = process_config["audio"]["sample_rate"]
#         self.max_wav_value = process_config["audio"]["max_wav_value"]
#         self.win_len = process_config["audio"]["win_len"]
#         self.max_len = process_config["audio"]["max_len"]
#         self.wavs = self.process_meta()
#
#         sr = process_config["audio"]["or_sample_rate"]
#         self.resample = torchaudio.transforms.Resample(sr, self.sample_rate)
#         self.sample_list = []
#         # 2.05s * 16000 = 32800 frames
#         # 0.5s * 16000 = 8000 frames
#         # Input audio length needs to be greater than 2.55s (32800+8000 frames)
#         min_length = 32800 + train_config["future_amt_waveform"]
#
#         # Randomly select half of the wav files
#         self.wavs = random.sample(self.wavs, len(self.wavs) // 2)
#
#         for idx in range(len(self.wavs)):
#             audio_name = self.wavs[idx]
#
#             wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
#
#             if wav.shape[1] > min_length:
#                 if wav.shape[1] > self.max_len:
#                     cuted_len = random.randint(5 * sr, self.max_len)
#                     wav = wav[:, :cuted_len]
#                 if sr != self.sample_rate:
#                     wav = self.resample(wav[0, :].view(1, -1))
#                 sample = {"matrix": wav, "sample_rate": sr, "name": audio_name}
#                 self.sample_list.append(sample)
#             else:
#                 print("The length of {} is shorter than 2.55s".format(audio_name))
#
#     def __len__(self):
#         return len(self.sample_list)
#
#     def __getitem__(self, idx):
#         return self.sample_list[idx]
#
#     def process_meta(self):
#         wavs = os.listdir(self.dataset_path)
#         return wavs


class wav_dataset(Dataset):
    def __init__(self, process_config, train_config, flag="train"):
        self.dataset_name = train_config["dataset"]
        raw_dataset_path = train_config["path"]["raw_path"]
        self.future_amt_waveform = train_config["future_amt_waveform"]
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()

        self.original_sample_rate = process_config["audio"]["or_sample_rate"]
        self.resample_needed = self.original_sample_rate != self.sample_rate

        if self.resample_needed:
            self.resample = torchaudio.transforms.Resample(
                self.original_sample_rate, self.sample_rate
            )

        # 2.05s * 16000 = 32800 frames
        # 0.5s * 16000 = 8000 frames
        # Input audio length needs to be greater than 2.55s (32800+8000 frames)
        min_length = 32800 + self.future_amt_waveform

        # Filter out files that are too short without loading them
        self.valid_wavs = []
        for audio_name in self.wavs:
            audio_path = os.path.join(self.dataset_path, audio_name)
            info = torchaudio.info(audio_path)
            num_frames = info.num_frames
            if num_frames > min_length:
                self.valid_wavs.append(audio_path)
            else:
                print("The length of {} is shorter than 2.55s".format(audio_name))

        # # Optionally, load only half of the dataset
        # self.valid_wavs = random.sample(self.valid_wavs, len(self.valid_wavs) // 2)

    def __len__(self):
        return len(self.valid_wavs)

    def __getitem__(self, idx):
        audio_name = self.valid_wavs[idx]
        audio_path = os.path.join(self.dataset_path, audio_name)
        wav, sr = torchaudio.load(audio_path)

        # Ensure the audio meets the minimum length requirement
        min_length = 32800 + self.future_amt_waveform
        if wav.shape[1] > min_length:
            # Cut to a random length if longer than max_len
            if wav.shape[1] > self.max_len:
                cuted_len = random.randint(5 * sr, self.max_len)
                wav = wav[:, :cuted_len]

            # Resample if necessary
            if self.resample_needed:
                wav = self.resample(wav[0, :].view(1, -1))

            random.seed(42)  # Choose any integer as your seed
            sample = {"matrix": wav, "sample_rate": sr, "name": audio_name}
            return sample
        else:
            raise ValueError(
                "The length of {} is shorter than 2.55s".format(audio_name)
            )

    def process_meta(self):
        return os.listdir(self.dataset_path)
