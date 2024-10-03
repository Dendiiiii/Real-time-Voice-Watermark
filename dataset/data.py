import torch
from torch.utils.data import Dataset
import torchaudio
import random
import os


def collate_fn(batch):
    # Find the maximum length in the batch for padding
    max_len = max([sample['matrix'].size(1) for sample in batch])

    # Pad each sample and update pad_num accordingly
    for sample in batch:
        original_len = sample['matrix'].size(1)
        pad_size = max_len - original_len
        sample['matrix'] = torch.nn.functional.pad(sample['matrix'], (0, pad_size))  # Pad with zeros
        sample['pad_num'] = pad_size  # Update the padding number

    # Combine all samples into a single batch dictionary
    batched_sample = {
        "matrix": torch.cat([sample['matrix'] for sample in batch], dim=0),
        "sample_rate": torch.tensor([sample['sample_rate'] for sample in batch]),
        "patch_num": torch.tensor([sample['patch_num'] for sample in batch]),
        "pad_num": torch.tensor([sample['pad_num'] for sample in batch]),
        "name": [sample['name'] for sample in batch]
    }
    return batched_sample


class wav_dataset(Dataset):
    def __init__(self, process_config, train_config, flag='train'):
        self.dataset_name = train_config["dataset"]
        raw_dataset_path = train_config["path"]["raw_path"]
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()

        sr = process_config["audio"]["or_sample_rate"]
        self.resample = torchaudio.transforms.Resample(sr, self.sample_rate)
        self.sample_list = []
        # 2.05s * 16000 = 32800 frames
        # 0.5s * 16000 = 8000 frames
        # Input audio length needs to be greater than 2.55s (32800+8000 frames)
        min_length = 32800+8000

        for idx in range(len(self.wavs)):
            audio_name = self.wavs[idx]

            wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))

            if wav.shape[1] > min_length:
                if wav.shape[1] > self.max_len:
                    cuted_len = random.randint(5*sr, self.max_len)
                    wav = wav[:, :cuted_len]
                if sr != self.sample_rate:
                    wav = self.resample(wav[0, :].view(1, -1))
                sample = {
                    "matrix": wav,
                    "sample_rate": sr,
                    "patch_num": 0,
                    "pad_num": 0,
                    "name": audio_name
                }
                self.sample_list.append(sample)
            else:
                print("The length of {} is shorter than 2.55s".format(audio_name))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        return self.sample_list[idx]

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs
