import typing as tp
import math
import numpy as np
import torch
import torch.nn as nn
from libs.modules.conv import DownConvBlock, StreamableConvTranspose1d
from libs.modules.lstm import StreamableLSTM


class SEANetResnetBlock(nn.Module):
    """Residual block from SEANet model

    Args:
        dim (int): Dimension of the input/output.
        kernel_size (list): List of kernel size for the convolution
        dilations (list): List of dilations for the convolution
        activation (str): Activation function.
        activation_params (dict): Parameters to provide the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide the underlying normalization used along with the convolution.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolution.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection
    """

    def __init__(
        self,
        dim: int,
        kernel_sizes: tp.List[int] = [3, 1],
        dilations: tp.List[int] = [1, 1],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = 'none',
        norm_params: tp.Dict[str, tp.Any] = {},
        causal: bool = False,
        pad_mode: str = "reflect",
        compress: int = 2,
        true_skip: bool = True
    ):
        super().__init__()
        assert len(kernel_sizes) == len(
            dilations
        ), "Number of kernel sizes should match number of dilations"
        act = getattr(nn, activation)
        hidden = dim // compress
        block = []
        for i, (kernel_size, dilation) in enumerate(zip(kernel_sizes, dilations)):
            in_chs = dim if i == 0 else hidden
            out_chs = dim if i == len(kernel_sizes) - 1 else hidden
            block += [
                act(**activation_params),
                StreamableConv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode
                )
            ]
            self.block = nn.Sequential(*block)
            self.shortcut: nn.Module
            if true_skip:
                self.shortcut = nn.Identity()
            else:
                self.shortcut = StreamableConv1d(
                    dim,
                    dim,
                    kernel_size=1,
                    norm=norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode
                )

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class SEANetEncoder(nn.Module):
    """SEANet encoder.

    Args:
        channels (int): Audio channels.
        dimension (int): Intermediate representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): kernel size and stride ratios. The encoder uses downsampling ratios instead of
            upsampling ratios, hence it will use the ratios in the reverse order to the ones specified here
            that must match the decoder order. We use the decoder order as some models may only employ the decoder.
        activation (str): Activation function.
        activation_params (dict): Parameters to provide to the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolution.
        residual_kernel_size (int): Kernel size for the residual layers.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution.
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the encoder, it corresponds to the N first blocks.
    """

    def __init__(
        self,
        channels: int = 2,
        dimension: int = 512,
        n_filters: int = 32,
        n_residual_layers: int = 1,
        # n_residual_layers: int = 3,
        # ratios: tp.List[int] = [1, 2],
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 5,
        last_kernel_size: int = (2, 2),
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        lstm: int = 0,
        disable_norm_outer_blocks: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.dimension = dimension
        self.n_filters = n_filters
        self.ratios = list(reversed(ratios))
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # first and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        assert (
                0 <= self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            DownConvBlock(
                channels,
                mult * n_filters,
                kernel_size,
                norm="none" if self.disable_norm_outer_blocks >= 1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            )
        ]

        # Downsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = "none" if self.disable_norm_outer_blocks >= i + 2 else norm
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        norm=block_norm,
                        norm_params=norm_params,
                        activation=activation,
                        activation_params=activation_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip,
                    )
                ]
            # Add downsampling layers
            model += [
                act(**activation_params),
                DownConvBlock(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=kernel_size,
                    stride=(2, 2),
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        model += [
            act(**activation_params),
            DownConvBlock(
                mult * n_filters,
                dimension,
                last_kernel_size,
                stride=(1, 1),
                dilation=(1, 1),
                norm=(
                    "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        signal_fft = self.stft(x)

        return self.model(signal_fft)


class SEANetEncoderKeepDimension(SEANetEncoder):
    """
    Similar architectures to the SEANet encoder but with an extra step that projects the output dimension
    to the same output dimension by repeating the sequential

    Args:
        SEANetEncoder (_type_): _description_
    """

    def __init__(self, *args, **kwargs):
        self.output_dim = kwargs.pop("output_dim")
        super().__init__(*args, **kwargs)
        # Adding a reverse convolution layer
        self.reverse_convolution = nn.ConvTranspose1d(
            in_channels=self.dimension,
            out_channels=self.output_dim,
            kernel_size=math.prod(self.ratios),
            stride=math.prod(self.ratios),
            padding=0
        )

    def forward(self, x):
        orig_nframes = x.shape[-1]
        x = self.model(x)
        x = self.reverse_convolution(x)
        # make change dim didn't change
        return x[:, :, :orig_nframes]


class SEANetDecoder(nn.Module):
    """SEANet decoder

    Args:
        channels (int): Audio channels.
        dimension (int): Intermedia representation dimension.
        n_filters (int): Base width for the model.
        n_residual_layers (int): nb of residual layers.
        ratios (Sequence[int]): Kernel size and stride ratios.
        activation (str): Activation function.
        activation_param (dict): Parameters to provide the activation function.
        final_activation (str): Final activation function after all the convolutions.
        final_activation_params (dict): Parameters to provikde the activation function.
        norm (str): Normalization method.
        norm_params (dict): Parameters to provide to the underlying normalization used along with the convolution.
        kernel_size (int): Kernel size for the initial convolution.
        last_kernel_size (int): Kernel size for the initial convolutions.
        residual_kernel_size (int): Kernel size for the residual layer.
        dilation_base (int): How much to increase the dilation with each layer.
        causal (bool): Whether to use fully causal convolution
        pad_mode (str): Padding mode for the convolutions.
        true_skip (bool): Whether to use true skip connection or a simple.
            (streamable) convolution as the skip connection in the residual network blocks.
        compress (int): Reduced dimensionality in residual branches (from Demucs v3).
        lstm (int): Number of LSTM layers at the end of the encoder.
        disable_norm_outer_blocks (int): Number of blocks for which we don't apply norm.
            For the decoder, it corresponds to the N last blocks.
        trim_right_ratio (float): Ratio for tri at the right of the transposed convolution under the causal setup.
            If equal to 1.0, it means that all the trimming is done at the right.
        """

    def __init__(
        self,
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 1, # 3,
        # ratios: tp.List[int] = [8, 5, 4, 2],
        ratios: tp.List[int] = [4],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
        final_activation: tp.Optional[str] = None,
        final_activation_params: tp.Optional[dict] = None,
        norm: str = "none",
        norm_params: tp.Dict[str, tp.Any] = {},
        kernel_size: int = 7,
        last_kernel_size: int = 7,
        residual_kernel_size: int = 3,
        dilation_base: int = 2,
        causal: bool = False,
        pad_mode: str = "reflect",
        true_skip: bool = True,
        compress: int = 2,
        lstm: int = 0,
        disable_norm_outer_blocks: int = 0,
        trim_right_ratio: float = 1.0,
        future_ts: int = 50
    ):
        super().__init__()
        self.dimension = dimension
        self.channels = channels
        self.n_filters = n_filters
        self.ratios = ratios
        del ratios
        self.n_residual_layers = n_residual_layers
        self.hop_length = np.prod(self.ratios)
        self.n_blocks = len(self.ratios) + 2  # First and last conv + residual blocks
        self.disable_norm_outer_blocks = disable_norm_outer_blocks
        self.future_ts = future_ts
        assert (
                0 <= self.disable_norm_outer_blocks <= self.n_blocks
        ), (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0"
        )

        act = getattr(nn, activation)
        mult = int(2 ** len(self.ratios))
        model: tp.List[nn.Module] = [
            StreamableConv1d(
                dimension,
                mult * n_filters,
                kernel_size,
                norm=(
                    "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode
            )
        ]

        if lstm:
            model += [StreamableLSTM(mult * n_filters, num_layers=lstm)]

        # Upsample to raw audio scale
        for i, ratio in enumerate(self.ratios):
            block_norm = (
                "none"
                if self.disable_norm_outer_blocks >= self.n_blocks - (i + 1)
                else norm
            )
            # Add upsampling layers
            model += [
                act(**activation_params),
                StreamableConvTranspose1d(
                    mult * n_filters,
                    mult * n_filters // 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    trim_right_ratio=trim_right_ratio

                )
            ]
            # Add residual layers
            for j in range(n_residual_layers):
                model += [
                    SEANetResnetBlock(
                        mult * n_filters // 2,
                        kernel_sizes=[residual_kernel_size, 1],
                        dilations=[dilation_base**j, 1],
                        activation=activation,
                        activation_params=activation_params,
                        norm=block_norm,
                        norm_params=norm_params,
                        causal=causal,
                        pad_mode=pad_mode,
                        compress=compress,
                        true_skip=true_skip
                    )
                ]

            mult //= 2

        # Add final layers
        model += [
            act(**activation_params),
            StreamableConv1d(
                n_filters,
                channels,
                last_kernel_size,
                norm="none" if self.disable_norm_outer_blocks >=1 else norm,
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode
            )
        ]
        # Add optional final activation to decoder (eg. tanh)
        if final_activation is not None:
            final_act = getattr(nn, final_activation)
            final_activation_params = final_activation_params or {}
            model += [final_act(**final_activation_params)]
        self.model = nn.Sequential(*model, nn.AdaptiveAvgPool1d(self.future_ts+1))

    def forward(self, z):
        y = self.model(z)
        return y


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

    def forward(self, x_spect, x=None):
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


class SimpleDetector(SimpleEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.reverse_convolution = nn.Sequential(nn.ConvTranspose1d(1, self.ch1, 1, 1),
                                                 nn.LeakyReLU())

    def forward(self, x_spect, x=None):
        length = x.shape[-1]
        kernel_size = x.shape[2] - 1
        stride = x.shape[2] // length
        x_spect = self.model(x_spect)
        x_spect = x_spect.reshape(x_spect.shape[0], 1, -1)
        print("x after reshape:", x_spect.size())
        self.reverse_convolution[0] = nn.ConvTranspose1d(1, self.ch1, kernel_size, stride)
        output = self.reverse_convolution(x_spect)
        print("output shape of reverse:", output.size())
        return output


# class SEANetEncoderKeepDimension(SEANetEncoder):
#     """
#     Similar architectures to the SEANet encoder but with an extra step that projects the output dimension
#     to the same output dimension by repeating the sequential
#
#     Args:
#         SEANetEncoder (_type_): _description_
#     """
#
#     def __init__(self, *args, **kwargs):
#         self.output_dim = kwargs.pop("output_dim")
#         super().__init__(*args, **kwargs)
#         # Adding a reverse convolution layer
#         self.reverse_convolution = nn.ConvTranspose1d(
#             in_channels=self.dimension,
#             out_channels=self.output_dim,
#             kernel_size=math.prod(self.ratios),
#             stride=math.prod(self.ratios),
#             padding=0
#         )
#
#     def forward(self, x):
#         orig_nframes = x.shape[-1]
#         x = self.model(x)
#         x = self.reverse_convolution(x)
#         # make change dim didn't change
#         return x[:, :, :orig_nframes]





