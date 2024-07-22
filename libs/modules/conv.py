import math
import typing as tp
import warnings

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
    ["none", "weight_norm", "spectral_norm", "time_group_norm"]
)


# def apply_parametrization_norm(module: nn.Module, norm: str = "none"):
#     assert norm in CONV_NORMALIZATIONS
#     if norm == "weight_norm":
#         return weight_norm(module)
#     elif norm == "spectral_norm":
#         return spectral_norm(module)
#     else:
#         # We already check in CONV_NORMALIZATION, so any other choice
#         # doesn't need reparameterization
#         return module
#
#
# def get_norm_module(
#         module: nn.Module, causal: bool = False, norm: str = "BatchNorm2d", **norm_kwargs
# ):
#     """Return the proper normalization module. If causal is True, this will ensure the returned
#     module is casual, or return an error if the normalization doesn't suppor causal evaluation.
#     """
#     assert norm in CONV_NORMALIZATIONS
#     if norm == "time_group_norm":
#         if causal:
#             raise ValueError("GroupNorm doesn't support causal evaluation.")
#         assert isinstance(module, nn.modules.conv._ConvNd)
#         return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
#     elif norm == "none":
#         return nn.Identity()
#     else:
#         return getattr(torch.nn, norm)(module.out_channels)
#
#
# def get_extra_padding_for_conv2d(
#         x: torch.Tensor, kernel_size: int, stride: int, padding_total: int = 0
# ) -> int:
#     """See 'pad_for_conv2d'"""
#     height = x.shape[-2]
#     width = x.shape[-1]
#     n_height_frames = (height - kernel_size + padding_total) / stride + 1
#     n_width_frames = (width - kernel_size + padding_total) / stride + 1
#     ideal_height = (math.ceil(n_height_frames) - 1) * stride + (kernel_size - padding_total)
#     ideal_width = (math.ceil(n_width_frames) - 1) * stride + (kernel_size - padding_total)
#
#     extra_padding_height = ideal_height - height
#     extra_padding_width = ideal_width - width
#     return extra_padding_width, extra_padding_height


# def pad2d(
#         x: torch.Tensor,
#         paddings: tp.Tuple[int, int, int, int],
#         mode: str = "reflect",
#         value: float = 0.0
# ):
#     """Tiny wrapper around F.apd, just to allow for reflect padding on small input.
#     If this is the case, we insert extra 0 padding to the right before the reflection happen.
#     """
#     height, width = x.shape[-2:]
#     padding_top, padding_bottom, padding_left, padding_right = paddings
#     assert (
#             padding_top >= 0 and padding_bottom >= 0 and padding_left >= 0 and padding_right >= 0
#     ), (padding_top, padding_bottom, padding_left, padding_right)
#     if mode == "reflect":
#         max_pad_vertical = max(padding_top, padding_bottom)
#         max_pad_horizontal = max(padding_left, padding_right)
#         extra_pad_vertical = 0
#         extra_pad_horizontal = 0
#
#         if height <= max_pad_vertical and width > max_pad_horizontal:
#             extra_pad_vertical = max_pad_vertical - height + 1
#             x = F.pad(x, (0, 0, 0, extra_pad_vertical))
#
#         elif width <= max_pad_horizontal and height > max_pad_vertical:
#             extra_pad_horizontal = max_pad_horizontal - width + 1
#             x = F.pad(x, (0, extra_pad_horizontal, 0, 0))
#
#         elif width <= max_pad_horizontal and height <= max_pad_vertical:
#             extra_pad_vertical = max_pad_vertical - height + 1
#             extra_pad_horizontal = max_pad_horizontal - width + 1
#             x = F.pad(x, (0, extra_pad_horizontal, 0, extra_pad_vertical))
#
#         padded = F.pad(x, paddings, mode, value)
#         end_height = padded.shape[-2] - extra_pad_vertical
#         end_width = padded.shape[-1] - extra_pad_horizontal
#
#         return padded[..., :end_height, :end_width]
#     else:
#         return F.pad(x, paddings, mode, value)
#
#
# def unpad2d(x: torch.Tensor, paddings: tp.Tuple[int, int, int, int]):
#     """remove padding from x, handling properly zero padding. Only for 2d"""
#     padding_left, padding_right, padding_top, padding_bottom = paddings
#     assert padding_top >= 0 and padding_bottom >= 0 and padding_left >= 0 and padding_right >= 0, (
#         padding_top, padding_bottom, padding_left, padding_right)
#     assert (padding_top + padding_bottom) <= x.shape[-2], \
#         f"Invalid top/bottom padding: {padding_top}/{padding_bottom}"
#     assert (padding_left + padding_right) <= x.shape[-1], \
#         f"Invalid left/right padding: {padding_left}/{padding_right}"
#     end_height = x.shape[-2] - padding_bottom
#     end_width = x.shape[-1] - padding_right
#     return x[..., padding_top:end_height, padding_left:end_width]
#
#
# class NormConv2d(nn.Module):
#     """Wrapper around Conv2d and normalization applied to this conv to provide a uniform interface across
#     normalization approaches.
#     """
#
#     def __init__(
#             self,
#             *args,
#             causal: bool = False,
#             norm: str = "none",
#             norm_kwargs: tp.Dict[str, tp.Any] = {},
#             **kwargs
#     ):
#         super().__init__()
#         self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
#         self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
#         self.norm_type = norm
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.norm(x)
#         return x
#
#
# class StreamableConv2d(nn.Module):
#     """Conv2d with some builtin handling of asymmetric or causal padding
#     and normalization.
#     """
#
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int,
#         stride: int = 1,
#         dilation: int = 1,
#         groups: int = 1,
#         bias: bool = True,
#         causal: bool = False,
#         norm: str = "none",
#         norm_kwargs: tp.Dict[str, tp.Any] = {},
#         pad_mode: str = "reflect",
#     ):
#         super().__init__()
#         # warn user on unusual setup between dilation and stride
#         if stride > 1 and dilation > 1:
#             warnings.warn(
#                 "StreamableConv1d has been initialized with stride > 1 and dilation > 1"
#                 f" (kernel_size={kernel_size} stride={stride}, dilation={dilation})."
#             )
#         self.conv = NormConv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             dilation=dilation,
#             groups=groups,
#             bias=bias,
#             causal=causal,
#             norm=norm,
#             norm_kwargs=norm_kwargs,
#         )
#         self.causal = causal
#         self.pad_mode = pad_mode
#
#     def forward(self, x):
#         kernel_size = self.conv.conv.kernel_size[0]
#         stride = self.conv.conv.stride[0]
#         dilation = self.conv.conv.dilation[0]
#         kernel_size = (
#             kernel_size - 1
#         ) * dilation + 1  # effective kernel size with dilation.
#         padding_total = kernel_size - stride
#         extra_padding_width, extra_padding_height = get_extra_padding_for_conv2d(
#             x, kernel_size, stride, padding_total
#         )
#         if self.causal:
#             # Left padding for causal
#             x = pad2d(x, (padding_total, extra_padding_width, padding_total, extra_padding_height), mode=self.pad_mode)
#         else:
#             # Asymmetric padding required for odd strides
#             padding_right = padding_total // 2
#             padding_left = padding_total - padding_right
#             padding_bottom = padding_total // 2
#             padding_top = padding_total - padding_bottom
#             x = pad2d(
#                 x, (padding_left, padding_right + extra_padding_width,
#                     padding_top, padding_bottom + extra_padding_height), mode=self.pad_mode
#             )
#         return self.conv(x)


# class NormConvTranspose2d(nn.Module):
#     """Wrapper around ConvTranspose2d and normalization applied to this conv
#     to provide a uniform interface across normalization approaches.
#     """
#
#     def __init__(
#         self,
#         *args,
#         norm: str = "none",
#         norm_kwargs: tp.Dict[str, tp.Any] = {},
#         **kwargs,
#     ):
#         super().__init__()
#         self.convtr = apply_parametrization_norm(
#             nn.ConvTranspose2d(*args, **kwargs), norm
#         )
#         self.norm = get_norm_module(self.convtr, causal=False, norm=norm, **norm_kwargs)
#
#     def forward(self, x):
#         x = self.convtr(x)
#         x = self.norm(x)
#         return x

# class StreamableConvTranspose2d(nn.Module):
#     """ConvTranspose2d with some builtin handling of asymmetric or causal padding
#     and normalization.
#     """
#
#     def __init__(
#         self,
#         in_channels: int,
#         out_channels: int,
#         kernel_size: int,
#         stride: int = 1,
#         causal: bool = False,
#         norm: str = "none",
#         trim_right_ratio: float = 1.0,
#         norm_kwargs: tp.Dict[str, tp.Any] = {},
#     ):
#         super().__init__()
#         self.convtr = NormConvTranspose2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride,
#             causal=causal,
#             norm=norm,
#             norm_kwargs=norm_kwargs,
#         )
#         self.causal = causal
#         self.trim_right_ratio = trim_right_ratio
#         assert (
#             self.causal or self.trim_right_ratio == 1.0
#         ), "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
#         assert 0.0 <= self.trim_right_ratio <= 1.0
#
#     def forward(self, x):
#         kernel_size = self.convtr.convtr.kernel_size[0]
#         stride = self.convtr.convtr.stride[0]
#         padding_total = kernel_size - stride
#
#         y = self.convtr(x)
#
#         # We will only trim fixed padding. Extra padding from `pad_for_conv2d` would be
#         # removed at the very end, when keeping only the right length for the output,
#         # as removing it here would require also passing the length at the matching layer
#         # in the encoder.
#         if self.causal:
#             # Trim the padding on the right according to the specified ratio
#             # if trim_right_ratio = 1.0, trim everything from right
#             padding_right = math.ceil(padding_total * self.trim_right_ratio)
#             padding_left = padding_total - padding_right
#             padding_top = math.ceil(padding_total * self.trim_right_ratio)
#             padding_bottom = padding_total - padding_top
#
#             y = unpad2d(y, (padding_left, padding_right, padding_top, padding_bottom))
#         else:
#             # Asymmetric padding required for odd strides
#             padding_right = padding_total // 2
#             padding_left = padding_total - padding_right
#             padding_bottom = padding_total // 2
#             padding_top = padding_total - padding_bottom
#
#             y = unpad2d(y, (padding_left, padding_right, padding_top, padding_bottom))
#         return y


