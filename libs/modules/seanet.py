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


# class SimpleDetector(nn.Module):
#     def __init__(self, n_fft=320, hop_length=160):
#         super(SimpleDetector, self).__init__()
#         self.n_fft = n_fft
#         self.hop_length = hop_length
#
#         # Convolutional layers with stride for downsampling
#         self.conv1 = nn.Conv2d(2, 16, kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1)
#
#         # Fully connected layers (sizes will be set dynamically)
#         self.fc1 = nn.Linear(1, 256)  # Placeholder sizes
#         self.fc2 = nn.Linear(256, 1)  # Placeholder sizes
#
#     def forward(self, x):
#         # x is of shape (b, length)
#         input_length = x.size(1)
#         window = torch.hann_window(self.n_fft).to(x.device)
#         # Apply STFT
#         x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False)
#         # x is now of shape (b, 2, freq_bins, time_frames)
#
#         # Permute to match conv layer expectations
#         x = x.permute(0, 3, 1, 2)  # (b, time_frames, 2, freq_bins)
#
#         # Apply convolutional layers
#         x = F.relu(self.conv1(x))  # (b, 16, freq_bins//2, time_frames//2)
#         x = F.relu(self.conv2(x))  # (b, 32, freq_bins//4, time_frames//4)
#         x = F.relu(self.conv3(x))  # (b, 64, freq_bins//8, time_frames//8)
#
#         # Calculate flattened size dynamically
#         flattened_size = x.size(1) * x.size(2) * x.size(3)
#
#         # Update fully connected layers with correct sizes
#         self.fc1 = nn.Linear(flattened_size, 256).to(x.device)
#         self.fc2 = nn.Linear(256, 2 * input_length).to(x.device)
#
#         # Flatten
#         x = x.view(x.size(0), -1)  # (b, flattened_size)
#
#         # Fully connected layers
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#
#         # Reshape to (b, 2, length)
#         x = x.view(x.size(0), 2, -1)
#
#         return x

# class SimpleDetector(nn.Module):
#     def __init__(
#             self,
#             channels: int = 2,
#             dimension: int = 128,
#             n_filters: int = 32,
#             n_residual_layers: int = 3,
#             ratios: tp.List[int] = [8, 5, 4, 2],
#             activation: str = "ELU",
#             activation_params: dict = {"alpha": 1.0},
#             norm: str = "none",
#             norm_params: tp.Dict[str, tp.Any] = {},
#             kernel_size: int = 7,
#             last_kernel_size: int = 3,
#             residual_kernel_size: int = 3,
#             dilation_base: int = 2,
#             causal: bool = False,
#             pad_mode: str = "reflect",
#             true_skip: bool = True,
#             compress: int = 2,
#             disable_norm_outer_blocks: int = 0
#     ):
#         super().__init__()
#         self.channels = channels
#         self.dimension = dimension
#         self.n_filters = n_filters
#         self.ratios = list(reversed(ratios))
#         del ratios
#         self.n_residual_layers = n_residual_layers
#         self.hop_length = np.prod(self.ratios)
#         self.n_blocks = len(self.ratios)  # + 2  # first and last conv + residual blocks
#         self.disable_norm_outer_blocks = disable_norm_outer_blocks
#         assert (
#                 0 <= self.disable_norm_outer_blocks <= self.n_blocks
#         ), (
#             "Number of blocks for which to disable norm is invalid."
#             "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0"
#         )
#
#         act = getattr(nn, activation)
#         mult = 1
#
#         model: tp.List[nn.Module] = [
#             DownConvBlock(
#                 channels,
#                 mult * n_filters,
#                 kernel_size,
#                 stride=1)
#         ]
#         # Downsampling layers
#         for i, ratio in enumerate(self.ratios):
#
#             model += [
#                 DownConvBlock(
#                     mult * n_filters,
#                     mult * n_filters * 2,
#                     kernel_size=ratio * 2,
#                     stride=ratio
#                 )
#             ]
#             mult *= 2
#
#         model += [
#             DownConvBlock(
#                 mult * n_filters,
#                 dimension,
#                 last_kernel_size,
#                 stride=1
#             )
#         ]
#
#         self.model = nn.Sequential(*model)
#         self.reverse_convolution = nn.ConvTranspose2d(
#             in_channels=self.dimension,
#             out_channels=2,
#             kernel_size=math.prod(self.ratios),
#             stride=math.prod(self.ratios),
#             padding=0
#         )
#
#     def forward(self, x):
#         print("x before detecor shape", x.size())
#         x = self.model(x)
#         print("x before reverse conv shape", x.size())
#         x = self.reverse_convolution(x)
#         print("x output shape", x.size())
#         return x


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


# class SimpleDetector(nn.Module):
#     def __init__(self):
#         super(SimpleDetector, self).__init__()
#         self.ch1 = 64
#         self.ch2 = 128
#         self.ch3 = 256
#
#         # Reverse order and channels of the encoder
#         model = [
#             nn.Sequential(UpConvBlock(2, self.ch3, (2, 2), (1, 1), (0, 0))),
#             nn.Sequential(UpConvBlock(self.ch3, self.ch2, 3, (2, 2), (1, 1)),
#                           UpConvBlock(self.ch2, self.ch2, 3, (2, 2), (1, 1)),
#                           UpConvBlock(self.ch2, self.ch3, 3, (2, 2), (1, 1)),
#                           UpConvBlock(self.ch3, self.ch2, 5, (2, 2), (1, 1))),
#             nn.Sequential(UpConvBlock(self.ch2, self.ch2, 5, (2, 2), (1, 1)),
#                           UpConvBlock(self.ch2, self.ch1, 5, (2, 2), (1, 1))),
#             nn.Sequential(UpConvBlock(self.ch1, 2, 5, (2, 2), (1, 1)))
#         ]
#
#         self.model = nn.Sequential(*model)
#
#     def forward(self, x_enc):
#         print("detector input shape", x_enc.size())
#         x_dec = self.model(x_enc)
#         print("detector output shape", x_dec.size())
#         return x_dec

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





