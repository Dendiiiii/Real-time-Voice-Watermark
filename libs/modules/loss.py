import numpy as np
import torch
import torch.nn as nn
from audiotools import AudioSignal


# Approximated Fletcher-Munson curve
def fletcher_munson_weights(freqs):
    # Ensure frequencies are at least 1e-6 to avoid division by zero
    freqs = freqs.clamp(min=1e-6)
    freq_ratio = freqs / 1000
    term1 = 3.64 * freq_ratio.pow(-0.8)
    term2 = -6.5 * torch.exp(-0.6 * (freq_ratio - 3.3).pow(2))
    term3 = 1e-3 * freq_ratio.pow(4)
    return term1 + term2 + term3


def perceptual_loss(watermark, sample_rate):

    # Verify input tensor
    if torch.isnan(watermark).any() or torch.isinf(watermark).any():
        raise ValueError("Input tensor contains NaNs or Infs.")

    # Compute the real FFT of the watermark over the last dimension
    try:
        print("FFT execution on GPU!")
        watermark_fft = torch.fft.rfft(watermark, dim=-1)
    except RuntimeError as e:
        print("FFT failed on GPU, attempting on CPU...")
        watermark_cpu = watermark.cpu()
        watermark_fft = torch.fft.rfft(watermark_cpu, dim=-1)
        watermark_fft = watermark_fft.to(watermark.device)

    # Generate frequencies corresponding to FFT components
    freqs = torch.fft.rfftfreq(watermark.size(-1), d=1 / sample_rate).to(
        watermark.device
    )

    # Calculate weights based on human ear sensitivity
    weights = fletcher_munson_weights(freqs)
    weights.clamp_(min=1e-6)

    # Apply weights to the magnitude of FFT components
    weighted_magnitude = torch.abs(watermark_fft)

    # Reshape weights to be broadcastable with weighted_magnitude
    weights = weights.view(*([1] * (weighted_magnitude.dim() - 1)), -1)

    # Multiply (broadcasting weights across batch and other dimensions)
    weighted_magnitude = weighted_magnitude * weights

    # Calculate the perceptual loss (sum over the frequency dimension)
    loss = weighted_magnitude.sum(dim=-1)

    # If there's a batch dimension, you may want to average over the batch
    loss = loss.mean()
    return torch.log1p(loss)


# The Total Variation Loss
def tv_loss(w_x):
    # w_x is the watermarked audio tensor
    # The size of the w_x should be (batch_size, sequence_length)

    # Calculate differences between adjacent samples
    diff = w_x[:, 1:] - w_x[:, :-1]

    # Return the sum of squared differences
    loss = torch.sum(torch.pow(diff, 2))

    return loss


def gradient_penalty_loss(w_x):
    # Calculate the difference between consecutive samples
    grad = w_x[:, 1:] - w_x[:, :-1]

    # Return the sum of absolute differences (L1 norm)
    loss = torch.sum(torch.abs(grad))

    return loss


def frequency_domain_loss(x, w_x):
    window = torch.hann_window(320).to(x.device)

    # Compute the STFT for both signals
    stft_original = torch.stft(x, n_fft=320, return_complex=True, window=window)
    stft_watermarked_x = torch.stft(w_x, n_fft=320, return_complex=True, window=window)

    # Calculate the difference in the frequency domain
    frequency_loss = torch.nn.functional.mse_loss(
        torch.abs(stft_original), torch.abs(stft_watermarked_x)
    )

    return frequency_loss


class Loss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce_loss = nn.BCELoss()
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

    def en_de_loss(
        self, x, w_x, wm, prob, labels, decoded_msg, message, sample_rate=16000
    ):
        # 从小数点到db level做mse, regularizer
        bce_loss = self.bce_loss(prob[:, 0, :], labels)
        decode_bce_loss = self.bce_loss(decoded_msg, message)
        l1_loss = self.l1_loss(w_x, x)
        l2_loss = self.l2_loss(w_x, x)
        hybrid_loss_value = 0  # self.alpha * l1_loss + self.beta * l2_loss
        percep_loss = 0  # perceptual_loss(wm, sample_rate)
        # tvl_loss = tv_loss(wm)*0.1
        # grad_penalty_loss = gradient_penalty_loss(wm)*0.001
        smoothness_loss = 0  # tvl_loss + grad_penalty_loss
        # freq_loss = 0
        freq_loss = frequency_domain_loss(x, w_x)

        loudness_loss = 0.0
        # try:
        #     loudness_loss = (
        #             AudioSignal(x.unsqueeze(1), sample_rate).loudness()
        #             - AudioSignal(wm.unsqueeze(1), sample_rate).loudness()
        #     ).mean()
        # except Exception as e:
        #     print(f"Loudness loss calculation error: {e}")
        #     loudness_loss = torch.tensor(0.0).to(x.device)

        return (
            hybrid_loss_value,  # No
            bce_loss,
            percep_loss * 0.035,  # No
            smoothness_loss,  # No
            freq_loss,
            decode_bce_loss,
            loudness_loss,  # No
        )
