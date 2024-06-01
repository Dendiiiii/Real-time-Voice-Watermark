import torch
import torch.nn as nn
import torchaudio.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.embedding_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def en_de_loss(self, x, w_x, wm, prob):
        embedding_loss = self.embedding_loss(x, w_x)
        # bce_loss = self.bce_loss(x, prob)
        loudness_loss = torch.mean((F.loudness(wm, 16000) - F.loudness(wm, 16000)) ** 2)

        return embedding_loss, loudness_loss  # , bce_loss
