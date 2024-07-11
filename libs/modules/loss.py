import torch
import torch.nn as nn
import torchaudio.functional as F


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.embedding_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

    def en_de_loss(self, x, w_x, wm, prob, labels):
        embedding_loss = self.embedding_loss(x, w_x)
        print("prob size:", prob.size())
        print("label size:", labels.type(torch.float32).size())
        bce_loss = self.bce_loss(prob[:, 0, :].unsqueeze(1), labels.type(torch.float32))
        loudness_loss = torch.mean((F.loudness(x, 16000) - F.loudness(wm, 16000)) ** 2)

        return embedding_loss, bce_loss
