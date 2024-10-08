import numpy as np
import torch
import torch.nn as nn


# Approximated Fletcher-Munson curve
def fletcher_munson_weights(freqs):
    freqs = np.maximum(freqs, 1e-6)
    return (
        3.64 * (freqs / 1000) ** -0.8
        - 6.5 * np.exp(-0.6 * (freqs / 1000 - 3.3) ** 2)
        + (10**-3) * (freqs / 1000) ** 4
    )


def perceptual_loss(watermark, sample_rate=16000):
    # Compute the FFT of the watermark
    watermark_fft = torch.fft.fft(watermark)

    # Generate frequencies corresponding to FFT components
    freqs = torch.fft.fftfreq(watermark_fft.size(-1), d=1 / sample_rate).to(
        watermark.device
    )

    # Calculate weights based on human ear sensitivity
    weights = torch.tensor(fletcher_munson_weights(freqs.cpu().numpy())).to(
        watermark.device
    )

    # Ensure weights are non-negative (if there are any negative weights, set them to a small positive value)
    weights = torch.clamp(weights, min=1e-6)

    # Apply weights to the magnitude of FFT components
    weighted_magnitude = torch.abs(watermark_fft) * weights

    # Calculate the perceptual loss (sum of weighted magnitudes)
    loss = torch.sum(weighted_magnitude)

    return torch.log(1 + loss)


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

    def en_de_loss(self, x, w_x, wm, prob, labels, decoded_msg, message):
        # 从小数点到db level做mse, regularizer
        bce_loss = self.bce_loss(prob[:, 0, :], labels)
        decode_bce_loss = self.bce_loss(decoded_msg, message)
        l1_loss = self.l1_loss(w_x, x)
        l2_loss = self.l2_loss(w_x, x)
        hybrid_loss_value = self.alpha * l1_loss + self.beta * l2_loss
        percep_loss = perceptual_loss(wm)
        # tvl_loss = tv_loss(wm)*0.1
        # grad_penalty_loss = gradient_penalty_loss(wm)*0.001
        smoothness_loss = 0  # tvl_loss + grad_penalty_loss
        freq_loss = frequency_domain_loss(x, w_x)

        return (
            hybrid_loss_value * 0,
            bce_loss,
            percep_loss * 0.035,
            smoothness_loss,
            freq_loss * 0,
            decode_bce_loss,
        )
