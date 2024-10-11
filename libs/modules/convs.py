import math
import typing as tp
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    # Old Pytorch
    from torch.nn.utils import weight_norm

CONV_NORMALIZATIONS = frozenset(
    ["none", "weight_norm", "spectral_norm", "time_group_norm", "batch_norm"]
)


def apply_parametrization_norm(module: nn.Module, norm: str = "none"):
    assert norm in CONV_NORMALIZATIONS
    if norm == "weight_norm":
        return weight_norm(module)
    elif norm == "spectral_norm":
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(
    module: nn.Module, causal: bool = False, norm: str = "none", **norm_kwargs
):
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == "time_group_norm":
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(
    x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
) -> int:
    """See `pad_for_conv1d`."""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(
    x: torch.Tensor,
    paddings: tp.Tuple[int, int],
    mode: str = "constant",
    value: float = 0.0,
):
    """Tiny wrapper around F.pad, just to allow for reflect padding on small input.
    If this is the case, we insert extra 0 padding to the right before the reflection happen.
    """
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == "reflect":
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(
        self,
        *args,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        **kwargs,
    ):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class StreamableConv1d(nn.Module):
    """Conv1d with some builtin handling of asymmetric or causal padding
    and normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        causal: bool = False,
        norm: str = "none",
        norm_kwargs: tp.Dict[str, tp.Any] = {},
        pad_mode: str = "reflect",
    ):
        super().__init__()
        # warn user on unusual setup between dilation and stride
        if stride > 1 and dilation > 1:
            warnings.warn(
                "StreamableConv1d has been initialized with stride > 1 and dilation > 1"
                f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
            )
        self.conv = NormConv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            causal=causal,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        self.causal = causal
        self.pad_mode = pad_mode

    def forward(self, x):
        B, C, T = x.shape
        kernel_size = self.conv.conv.kernel_size[0]
        stride = self.conv.conv.stride[0]
        dilation = self.conv.conv.dilation[0]
        kernel_size = (
            kernel_size - 1
        ) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(
            x, kernel_size, stride, padding_total
        )
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(
                x, (padding_left, padding_right + extra_padding), mode=self.pad_mode
            )
        return self.conv(x)


class DownConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation=1,
        norm_fn="bn",
        act="prelu",
        pad=None,
    ):
        super(DownConvBlock, self).__init__()
        if pad is None:
            pad = (kernel_size - 1) // 2 * dilation
        block = [
            nn.ReflectionPad2d(pad),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                0,
                dilation,
                bias=norm_fn is None,
            ),
        ]
        if norm_fn == "bn":
            block.append(nn.BatchNorm2d(out_channels))
        elif act == "prelu":
            block.append(nn.PReLU())
        elif act == "lrelu":
            block.append(nn.LeakyReLU())

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UpConvBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding=0, dilation=1
    ):
        super(UpConvBlock, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            ),
            nn.LeakyReLU(),
        )

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


class SimpleEncoder(nn.Module):
    def __init__(self, dimension: int = 512):
        super(SimpleEncoder, self).__init__()
        self.ch1 = 64
        self.ch2 = 128
        self.ch3 = 256
        self.dimension = dimension

        model = [
            nn.Sequential(DownConvBlock(2, self.ch1, 5, (2, 2))),
            nn.Sequential(
                DownConvBlock(self.ch1, self.ch2, 5, (2, 2)),
                DownConvBlock(self.ch2, self.ch2, 5, (2, 2)),
            ),
            nn.Sequential(
                DownConvBlock(self.ch2, self.ch3, 5, (2, 2)),
                DownConvBlock(self.ch3, self.ch2, 3, (2, 2)),
                DownConvBlock(self.ch2, self.ch2, 3, (2, 2)),
                DownConvBlock(self.ch2, self.ch3, 3, (2, 2)),
            ),
            nn.Sequential(
                nn.Conv2d(self.ch3, self.dimension, (2, 2), (1, 1), 0, (1, 1)),
                nn.LeakyReLU(),
            ),
        ]

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
        self.linear = nn.Sequential(
            nn.Linear(4117, 8000), nn.Tanh()
        )  # 8000 samples = 0.5s

    def forward(self, x):
        up1 = self.up1(x)
        up2 = self.up2(up1)
        up3 = self.up3(up2)
        up4 = self.up4(up3)
        up5 = self.linear(up4)

        return up5


class SimpleDetector(nn.Module):
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
        channels: int = 1,
        dimension: int = 128,
        n_filters: int = 32,
        n_residual_layers: int = 3,
        ratios: tp.List[int] = [8, 5, 4, 2],
        activation: str = "ELU",
        activation_params: dict = {"alpha": 1.0},
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
        assert 0 <= self.disable_norm_outer_blocks <= self.n_blocks, (
            "Number of blocks for which to disable norm is invalid."
            "It should be lower or equal to the actual number of blocks in the network and greater or equal to 0."
        )

        act = getattr(nn, activation)
        mult = 1
        model: tp.List[nn.Module] = [
            StreamableConv1d(
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

            # Add downsampling layers
            model += [
                act(**activation_params),
                StreamableConv1d(
                    mult * n_filters,
                    mult * n_filters * 2,
                    kernel_size=ratio * 2,
                    stride=ratio,
                    norm=block_norm,
                    norm_kwargs=norm_params,
                    causal=causal,
                    pad_mode=pad_mode,
                ),
            ]
            mult *= 2

        model += [
            act(**activation_params),
            StreamableConv1d(
                mult * n_filters,
                dimension,
                last_kernel_size,
                norm=(
                    "none" if self.disable_norm_outer_blocks == self.n_blocks else norm
                ),
                norm_kwargs=norm_params,
                causal=causal,
                pad_mode=pad_mode,
            ),
        ]

        self.model = nn.Sequential(*model)

        self.reverse_convolution = nn.ConvTranspose1d(
            in_channels=self.dimension,
            out_channels=32,
            kernel_size=math.prod(self.ratios),
            stride=math.prod(self.ratios),
            padding=0,
        )

    def forward(self, x):
        orig_nframes = x.shape[-1]
        x = self.model(x.unsqueeze(1))
        x = self.reverse_convolution(x)
        # Make sure dim doesn't change
        return x[..., :orig_nframes]
