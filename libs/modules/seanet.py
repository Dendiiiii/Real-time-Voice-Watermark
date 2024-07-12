import typing as tp
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.modules.conv import DownConvBlock, StreamableConvTranspose1d
from libs.modules.lstm import StreamableLSTM


class DownConvBlock_new(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1,
                 norm_fn='bn',
                 act='prelu'):
        super(DownConvBlock_new, self).__init__()
        pad = (kernel_size - 1) // 2 * dilation
        block = []
        block.append(nn.ReflectionPad2d(pad))
        block.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, 0, dilation, bias=norm_fn is None))
        if norm_fn == "bn":
            block.append(nn.BatchNorm2d(out_channels))
        elif act == 'prelu':
            block.append(nn.PReLU())
        elif act == 'lrelu':
            block.append(nn.LeakyReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class SimpleEncoder(nn.Module):
    def __init__(self, dimension: int = 512):
        super(SimpleEncoder, self).__init__()
        self.ch1 = 64
        self.ch2 = 128
        self.ch3 = 256
        self.dimension = dimension

        model = [
            nn.Sequential(DownConvBlock_new(2, self.ch1, 5, (2, 2))),
            nn.Sequential(DownConvBlock_new(self.ch1, self.ch2, 5, (2, 2)),
                          DownConvBlock_new(self.ch2, self.ch2, 5, (2, 2))),
            nn.Sequential(DownConvBlock_new(self.ch2, self.ch3, 5, (2, 2)),
                          DownConvBlock_new(self.ch3, self.ch2, 3, (2, 2)),
                          DownConvBlock_new(self.ch2, self.ch2, 3, (2, 2)),
                          DownConvBlock_new(self.ch2, self.ch3, 3, (2, 2)),),
            nn.Sequential(nn.Conv2d(self.ch3, self.dimension, (2, 2), (1, 1), 0, (1, 1)),
                          nn.LeakyReLU())]

        self.model = nn.Sequential(*model)

    def forward(self, x_spect):
        x_spect = self.model(x_spect)
        return x_spect


class SimpleDecoder(nn.Module):
    def __init__(self):
        super(SimpleDecoder, self).__init__()
        self.up1 = nn.Sequential(nn.ConvTranspose1d(1, 64, 1, 1), nn.LeakyReLU())
        self.up2 = nn.Sequential(nn.ConvTranspose1d(64, 32, 5, 2), nn.LeakyReLU())
        self.up3 = nn.Sequential(nn.ConvTranspose1d(32, 16, 5, 2), nn.LeakyReLU())
        self.up4 = nn.Sequential(nn.ConvTranspose1d(16, 1, 5, 2), nn.Tanh())
        self.linear = nn.Sequential(nn.Linear(4117, 8000), nn.Tanh())  # 8000 samples = 0.5s

    def forward(self, x):
        up1 = self.up1(x)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        up4 = self.up4(up3)
        up5 = self.linear(up4)
        return up5


class SimpleDetector(nn.Module):
    def __init__(self, n_fft=320, hop_length=160):
        super(SimpleDetector, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Convolutional layers with stride for downsampling
        self.conv1 = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)

        # Fully connected layers (sizes will be set dynamically)
        self.fc1 = nn.Linear(1, 256)  # Placeholder sizes
        self.fc2 = nn.Linear(256, 1)  # Placeholder sizes

    def forward(self, x):
        # x is of shape (b, length)
        input_length = x.size(1)
        window = torch.hann_window(self.n_fft).to(x.device)
        # Apply STFT
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
        # x is now of shape (b, 2, freq_bins, time_frames)

        # Permute to match conv layer expectations
        x = x.permute(0, 3, 1, 2)  # (b, time_frames, 2, freq_bins)

        # Apply convolutional layers
        x = F.relu(self.conv1(x))  # (b, 16, freq_bins//2, time_frames//2)
        x = F.relu(self.conv2(x))  # (b, 32, freq_bins//4, time_frames//4)
        x = F.relu(self.conv3(x))  # (b, 64, freq_bins//8, time_frames//8)

        # Calculate flattened size dynamically
        flattened_size = x.size(1) * x.size(2) * x.size(3)

        # Update fully connected layers with correct sizes
        self.fc1 = nn.Linear(flattened_size, 256).to(x.device)
        self.fc2 = nn.Linear(256, 2 * input_length).to(x.device)

        # Flatten
        x = x.view(x.size(0), -1)  # (b, flattened_size)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        # Reshape to (b, 2, length)
        x = x.view(x.size(0), 2, -1)

        return x







