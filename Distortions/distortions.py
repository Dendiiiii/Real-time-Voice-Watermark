import torch
import random
import torch.nn as nn
import pdb
import numpy as np
import julius
import kornia


SAMPLE_RATE = 22050
device = torch.device("cudd:0" if torch.cuda.is_available() else "cpu")


class distortion(nn.Module):
    def __init__(self, process_config):
        super(distortion, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.resample_kernel_1 = julius.ResampleFrac(SAMPLE_RATE, 16000).to(self.device)
        self.resample_kernel_1_re = julius.ResampleFrac(16000, SAMPLE_RATE).to(self.device)
        self.resample_kernel_2 = julius.ResampleFrac(SAMPLE_RATE, 8000).to(self.device)
        self.resample_kernel_2_re = julius.ResampleFrac(8000, SAMPLE_RATE).to(self.device)
        # self.augment = Compose([Mp3Compression(p=1.0, min_bitrate=64, max_bitrate=64)])
        self.band_lowpass = julius.LowPassFilter(2000/SAMPLE_RATE).to(self.device)
        self.band_highpass = julius.HighPassFilter(500/SAMPLE_RATE).to(self.device)
        # self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"]).to(self.device)

    def none(self, x):
        return x

    def crop(self, x):
        length = x.shape[-1]
        if length > 18000:
            start = random.randint(0, 1000)
            end = random.randint(1, 1000)
            y = x[:, :, start:0-end]
        else:
            y = x
        return y

    def crop2(self, x):
        length = x.shape[-1]
        if length > 18000:
            cut_len = int(length * 0.1)  # cut 10% off  #length = 100, cut_len = 10
            start = random.randint(0, cut_len-1)  # start=8
            end = cut_len - start  # end = 10 - 8 = 2
            y = x[:, :, start:0-end]  # x[:, :, 8:0-2] = x[:, :, 8:-2]
        else:
            y = x
        return y

    def crop_front(self, x, cut_ratio=10):
        cut_len = int(x.shape[-1] * (cut_ratio/100))
        ret = x[:, :, cut_len:]
        return ret

    def resample(self, x):
        return x

    def crop_middle(self, x, cut_ratio=10):
        cut_len = int(x.shape[-1] * (cut_ratio/100))
        begin = int((x.shape[-1] - cut_len) / 2)
        end = begin + cut_len
        ret = torch.cat([x[:, :, :begin], x[:, :, end:]], dim=-1)
        return ret

    def crop_back(self, x, cut_ratio=10):
        cut_len = int(x.shape[-1] * (cut_ratio/100))
        begin = int((x.shape[-1] - cut_len))
        ret = x[:, :, :begin]
        return ret

    def resample1(self, x):
        y = self.resample_kernel_1_re(self.resample_kernel_1(x))
        return y

    def resample2(self, x):
        y = self.resample_kernel_2_re(self.resample_kernel_2(x))
        return y

    def white_noise(self, y, ratio): # SNR = 10log(ps/pn)
        SNR = ratio
        mean = 0.
        RMS_s = torch.sqrt(torch.mean(y**2, dim=-1))
        RMS_n = torch.sqrt(RMS_s**2/(pow(10, SNR/20)))
        for i in range(y.shape[0]):
            noise = torch.normal(mean, float(RMS_n[i][0]), size=(1, y.shape[2]))
            if i == 0:
                batch_noise = noise
            else:
                batch_noise = torch.cat((batch_noise, noise), dim=0)
        batch_noise = batch_noise.unsqueeze(1).to(self.device)
        signal_edit = y + batch_noise
        return signal_edit

    def change_top(self, y, ratio=50):
        y = y * ratio/100
        return y

    def mp3(self, y, ratio=64):
        # self.augment = Compose([Mp3Compression(p=1.0, min_bitrate=ratio, max_bitrate=ratio)])
        f = []
        a = y.cpu().detach().numpy()
        for i in a:
            f.append(torch.Tensor(self.augment(i, sample_rate=SAMPLE_RATE)))
        f = torch.cat(f, dim=0).unsqueeze(1).to(self.device)
        return f

    def recount(self, y):
        y2 = torch.tensor((np.array(y.cpu().squeeze(0).data.numpy()+1)/2*(2**8-1)).astype(np.uint8)) / (2**8-1) * 2 - 1
        y2 = y2.to(self.device)
        y = y + (y2 - y).detach()
        return y

    def medfilt(self, y, ratio=3):
        y = kornia.filters.median_blur(y.unsqueeze(1), (1, ratio)).squeeze(1)
        return y

    def low_band_pass(self, y):
        y = self.band_lowpass(y)
        return y

    def high_band_pass(self, y):
        y = self.band_highpass(y)
        return y

    def modify_mel(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        spect = spect * ratio/100
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def crop_mel_front(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(ratio/100))
        spect = spect*(torch.cat([torch.zeros(_, cut_len, time_len), torch.ones(_, fre_len-cut_len, time_len)], dim=1).to(self.device))
        return spect

    def crop_mel_back(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(ratio*100))
        spect = spect*(torch.cat([torch.ones(_, fre_len-cut_len, time_len), torch.zeros(_, cut_len, time_len)], dim=1).to(self.device))
        return spect

    def crop_mel_wave_front(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(ratio*100))
        spect = spect*(torch.cat([torch.zeros(_, cut_len, time_len), torch.ones(_, fre_len-cut_len, time_len)], dim=1).to(self.device))
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def crop_mel_wave_back(self, y, ratio=50):
        num_samples = y.shape[2]
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(ratio*100))
        spect = spect*(torch.cat([torch.ones(_, fre_len-cut_len, time_len), torch.zeors(_, cut_len, time_len)], dim=1).to(self.device))
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))
        return y

    def crop_mel_position(self, y, ratio=1):
        assert 1 <= ratio <= 10, "a must be an integer between 1 and 10"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(1/10))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        return 0

    def crop_mel_wave_position(self, y, ratio=1):
        num_samples = y.shape[2]
        assert 1 <= ratio <= 10, "a must be an integer between 1 and 10"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(1/10))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))

    def crop_mel_position_5(self, y, ratio=1):
        assert 1 <= ratio <= 20, "a must be an integer between 1 and 10"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(1/20))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        return 0

    def crop_mel_wave_position_5(self, y, ratio=1):
        num_samples = y.shape[2]
        assert 1 <= ratio <= 20, "a must be an integer between 1 and 10"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(1/20))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))

    def crop_mel_position_20(self, y, ratio=1):
        assert 1 <= ratio <= 5, "a must be an integer between 1 and 10"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(1/5))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        return 0

    def crop_mel_wave_position_20(self, y, ratio=1):
        num_samples = y.shape[2]
        assert 1 <= ratio <= 5, "a must be an integer between 1 and 10"
        spect, phase = self.stft.transform(y)
        _, fre_len, time_len = spect.shape
        cut_len = int(fre_len*(1/5))
        left, right = (ratio - 1) * cut_len, ratio * cut_len
        spect[:, left:right, :] = 0
        self.stft.num_samples = num_samples
        y = self.stft.inverse(spect.squeeze(1), phase.squeeze(1))

    def forward(self, x, attack_choice=1, ratio=10):
        attack_functions = {
            0: lambda x: self.none(x),
            1: lambda x: self.crop(x),
            2: lambda x: self.crop2(x),
            3: lambda x: self.resampe(x),
            4: lambda x: self.crop_front(x, ratio),       # Cropping front
            5: lambda x: self.crop_middle(x, ratio),      # Cropping middle
            6: lambda x: self.crop_back(x, ratio),        # Cropping behind
            7: lambda x: self.resample1(x),               # Resampling 16KHz
            8: lambda x: self.resample2(x),               # Resampling 8KHz
            9: lambda x: self.white_noise(x, ratio),      # Gaussian Noise with SNR ratio / 2 dB
            10: lambda x: self.change_top(x, ratio),      # Amplitude Scaling ratio%
            11: lambda x: self.mp3(x, ratio),             # MP3 Compression ratio Kbps
            12: lambda x: self.recount(x),                # Recount 8 bps
            13: lambda x: self.medfilt(x, ratio),         # Median Filtering with ratio samples as window
            14: lambda x: self.low_band_pass(x),          # Low Pass Filtering 2000 Hz
            15: lambda x: self.high_band_pass(x),         # High Pass Filtering 500 Hz
            16: lambda x: self.modify_mel(x, ratio),      # don't need
            17: lambda x: self.crop_mel_front(x, ratio),  # don't need
            18: lambda x: self.crop_mel_back(x, ratio),   # don't need
            19: lambda x: self.crop_mel_wave_front(x, ratio),       # don't need
            20: lambda x: self.crop_mel_wave_back(x, ratio),        # mask from top with ratio "ratio" and transform back to wav
            21: lambda x: self.crop_mel_position(x, ratio),         # mask 10% at position "ratio"
            22: lambda x: self.crop_mel_wave_position(x, ratio),    # mask 10% at positon "ratio and transform back to wav
            23: lambda x: self.crop_mel_position_5(x, ratio),       # mask 5% at position "ratio"
            24: lambda x: self.crop_mel_wave_position_5(x, ratio),  # mask 5% at position "ratio" and transform back to wav
            25: lambda x: self.crop_mel_position_20(x, ratio),      # mask 20% at position "ratio"
            26: lambda x: self.crop_mel_wave_position_20(x, ratio)  # mask 20% at position "ratio" and transform back to wav
        }

        x = x.clamp(-1, 1)
        y = attack_choice[attack_choice](x)
        y = y.clamp(-1, 1)
        return y























