import torch
import torch.nn as nn
import torchaudio
import pyloudnorm as sf
import torchaudio.functional as F


class TFLoudnessLoss(nn.Module):
    def __init__(self):
        super(TFLoudnessLoss, self).__init__()
    def forward(self, signal, watermark, sample_rate):
        meter = pyln.Meter(self.sample_rate)
        signal_loudness = meter.integrated_loudness(signal)
        watermark_loudness = meter.integrated_loudness(watermark)
        loudness_loss = watermark_loudness - signal_loudness
        return loudness_loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.loudness_loss = TFLoudnessLoss()

    def en_de_loss(self, x, w_x, wm, prob, labels):
        # 从小数点到db level做mse, regularizer
        bce_loss = self.bce_loss(prob[:, 0, :], labels.type(torch.float32))
        loudness_loss = self.loudness_loss(x, wm, 16000)
        return loudness_loss, bce_loss
