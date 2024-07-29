import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F


class TFLoudnessLoss(nn.Module):
    def __init__(self, B, W, r, sample_rate):
        super(TFLoudnessLoss, self).__init__()
        self.B = B  # number of frequency bands
        self.W = W  # window size
        self.r = r  # overlap ratio
        self.sample_rate = sample_rate
        self.cutoff_freqs = [(i * sample_rate / (2 * B), (i + 1) * sample_rate / (2 * B)) for i in range(B)]

    def forward(self, signal, watermark):
        losses = []
        for low_cut, high_cut in self.cutoff_freqs:
            # Apply bandpass filtering
            band_signal = torchaudio.functional.highpass_biquad(signal, self.sample_rate, low_cut)
            band_signal = torchaudio.functional.lowpass_biquad(band_signal, self.sample_rate, high_cut)
            band_watermark = torchaudio.functional.highpass_biquad(watermark, self.sample_rate, low_cut)
            band_watermark = torchaudio.functional.lowpass_biquad(band_watermark, self.sample_rate, high_cut)

            # Segment the band signals
            segments_signal = band_signal.unfold(1, self.W, self.W - int(self.r * self.W))
            segments_watermark = band_watermark.unfold(1, self.W, self.W - int(self.r * self.W))

            # Calculate loudness for each segment
            loudness_signal = self.calculate_loudness(segments_signal)
            loudness_watermark = self.calculate_loudness(segments_watermark)

            # Compute loudness difference and mean loss per band
            loss = torch.mean(torch.abs(loudness_watermark - loudness_signal))
            losses.append(loss)

        # Mean loss across all bands
        return torch.mean(torch.stack(losses))

    def calculate_loudness(self, segments):
        # Example coefficients for a simple biquad peaking EQ (Placeholders)
        b0 = 1.0
        b1 = -2.0
        b2 = 1.0
        a0 = 1.0
        a1 = -1.8
        a2 = 0.81

        # Apply the filter to each segment
        weighted_segments = torchaudio.functional.biquad(
            segments, b0, b1, b2, a0, a1, a2)

        # Calculate energy
        energy = torch.mean(weighted_segments ** 2, dim=-1)

        # Compute loudness
        loudness = 10 * torch.log10(energy + 1e-8)  # adding epsilon to avoid log(0)
        return loudness


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.loudness_loss = TFLoudnessLoss(5, 400, 0.5, 16000)

    def en_de_loss(self, x, w_x, wm, prob, labels):
        # 从小数点到db level做mse, regularizer
        bce_loss = self.bce_loss(prob[:, 0, :], labels.type(torch.float32))
        loudness_loss = self.loudness_loss(x, wm)
        return loudness_loss, bce_loss
