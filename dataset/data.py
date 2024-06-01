import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import random
import os


def collate_fn(batch):
    # Extract all audio tensors and their respective length
    audios = [item['matrix'].squeeze(0) for item in batch]
    lengths = [audio.shape[0] for audio in audios]  # audio shape is (1, len)
    # Pad the audio sequence so that all are of the same length
    audio_padded = pad_sequence(audios, batch_first=True, padding_value=0)  # Padding value can be 0 for waveform

    # Prepare other data like sample_rate, name etc., if you need to use them later
    sample_rates = [item["sample_rate"] for item in batch]
    names = [item["name"] for item in batch]

    # Create a dictionary or any structure that suits downstream processing
    batched_sample = {
        "matrix": audio_padded,
        "lengths": lengths,
        "sample_rates": sample_rates,
        "names": names
    }

    return batched_sample


class oned_dataset(Dataset):
    def __init__(self, process_config, train_config, flag='train'):
        self.dataset_name = train_config["dataset"]
        raw_dataset_path = train_config["path"]["raw_path"]
        self.dataset_path = os.path.join(raw_dataset_path, flag)
        self.sample_rate = process_config["audio"]["sample_rate"]
        self.max_wav_value = process_config["audio"]["max_wav_value"]
        self.win_len = process_config["audio"]["win_len"]
        self.max_len = process_config["audio"]["max_len"]
        self.wavs = self.process_meta()

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        audio_name = self.wavs[idx]
        wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
        wav = wav[:,:self.max_len]
        sample = {
            "matrix": wav,
            "sample_rate": sr,
            "patch_num": 0,
            "pad_num": 0,
            "name": audio_name
        }
        return sample

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs


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
        for idx in range(len(self.wavs)):
            audio_name = self.wavs[idx]
            wav, sr = torchaudio.load(os.path.join(self.dataset_path, audio_name))
            if wav.shape[1] > self.max_len:
                cuted_len = random.randint(5*sr, self.max_len)
                wav = wav[:, :cuted_len]
            wav = self.resample(wav[0, :].view(1, -1))
            sample = {
                "matrix": wav,
                "sample_rate": sr,
                "patch_num": 0,
                "pad_num": 0,
                "name": audio_name
            }
            self.sample_list.append(sample)

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx):
        return self.sample_list[idx]

    def process_meta(self):
        wavs = os.listdir(self.dataset_path)
        return wavs
