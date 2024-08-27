import numpy as np
import torch
import torch.nn as nn
import torchaudio
import pyloudnorm as pyln
import torchaudio.functional as F


# Approximated Fletcher-Munson curve
def fletcher_munson_weights(freqs):
    freqs = np.maximum(freqs, 1e-6)
    return 3.64 * (freqs/1000)**-0.8 - 6.5 * np.exp(-0.6 * (freqs/1000 - 3.3)**2) + (10**-3) * (freqs/1000)**4


def perceptual_loss(watermark, sample_rate=16000):
    # Compute the FFT of the watermark
    watermark_fft = torch.fft.fft(watermark)

    # Generate frequencies corresponding to FFT components
    freqs = torch.fft.fftfreq(watermark_fft.size(-1), d=1/sample_rate).to(watermark.device)

    # Calculate weights based on human ear sensitivity
    weights = torch.tensor(fletcher_munson_weights(freqs.cpu().numpy())).to(watermark.device)

    # Ensure weights are non-negative (if there are any negative weights, set them to a small positive value)
    weights = torch.clamp(weights, min=1e-6)

    # Apply weights to the magnitude of FFT components
    weighted_magnitude = torch.abs(watermark_fft) * weights

    # Calculate the perceptual loss (sum of weighted magnitudes)
    loss = torch.sum(weighted_magnitude)

    return loss


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        # self.loudness_loss = TFLoudnessLoss()
        self.l1_loss = nn.L1Loss()

    def en_de_loss(self, x, w_x, wm, prob, labels):
        # 从小数点到db level做mse, regularizer
        bce_loss = self.bce_loss(prob[:, 0, :], labels.type(torch.float32))
        # loudness_loss = self.loudness_loss(x, wm, 16000)
        l1_loss = self.l1_loss(wm, torch.zeros_like(wm))
        percep_loss = perceptual_loss(wm)
        return l1_loss, bce_loss, percep_loss
