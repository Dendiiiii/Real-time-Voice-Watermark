import typing as tp
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from libs.modules.conv import *


class DownConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 dilation=1,
                 norm_fn='bn',
                 act='prelu'):
        super(DownConvBlock, self).__init__()
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
            nn.Sequential(DownConvBlock(2, self.ch1, 5, (2, 2))),
            nn.Sequential(DownConvBlock(self.ch1, self.ch2, 5, (2, 2)),
                          DownConvBlock(self.ch2, self.ch2, 5, (2, 2))),
            nn.Sequential(DownConvBlock(self.ch2, self.ch3, 5, (2, 2)),
                          DownConvBlock(self.ch3, self.ch2, 3, (2, 2)),
                          DownConvBlock(self.ch2, self.ch2, 3, (2, 2)),
                          DownConvBlock(self.ch2, self.ch3, 3, (2, 2)),),
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


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                                   padding=padding, dilation=dilation), nn.LeakyReLU())

        # self.double_conv = nn.Sequential(
        #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     nn.ReLU(inplace=True)
        # )

    def forward(self, x):
        x = self.up(x)
        # x = self.double_conv(x)
        return x


class SimpleDetector(SimpleEncoder):
    def __init__(self, n_fft=320, hop_length=160, *args, **kwargs):
        super().__init__()
        # Reverse order and channels of the encoder
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.detector = nn.Sequential(
            UpConvBlock(self.dimension, self.ch3, 5, 5, (0, 0)),
            UpConvBlock(self.ch3, self.ch2, 2, 2, (0, 0)),
            UpConvBlock(self.ch2, self.ch2, 4, 4, (0, 0)),
            UpConvBlock(self.ch2, self.ch2, 4, 4, (0, 0)),
            UpConvBlock(self.ch2, 2, 2, 1, (0, 0)))

    def istft(self, x):
        window = torch.hann_window(self.n_fft).to(x.device)
        x = torch.view_as_complex(x.permute(0, 2, 3, 1).contiguous())
        tmp = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window)
        return tmp

    def forward(self, x):
        x = self.model(x)
        x = self.detector(x)
        x = self.istft(x).unsqueeze(1)
        return x





