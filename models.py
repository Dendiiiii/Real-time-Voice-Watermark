import logging
from typing import Optional, Tuple
import torch.nn.functional as F

from torch import Tensor
from torch.nn import CTCLoss
import julius
import torch
from libs.modules.seanet import SEANetEncoderKeepDimension

logger = logging.getLogger("VoiceWatermark")


class WatermarkModel(torch.nn.Module):
    """
    Generate watermarking for a given audio signal
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        future_ts: float = 50,
        future: bool = True,
        wandb: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.CEloss = torch.nn.CrossEntropyLoss()
        # self.CTCloss = CTCLoss(blank=0, reduction='sum', zero_infinity=True)
        self.future_ts = future_ts
        self.future = future
        self.wandb = wandb

    def get_watermark(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = 16000,
    ) -> torch.Tensor:
        """
        Get the watermark from an audio tensor.
        Args:
            x: Audio signal, size: batch x frames
            sample_rate: The sample rate of input audio (default 16khz as
                currently supported by the main model)
        """
        length = x.size(-1)
        if sample_rate is None:
            logger.warning("No sample rate input, setting it to be 16khz")
            sample_rate = 16_000
        assert sample_rate
        if sample_rate != 16000:
            x = julius.resample_frac(x, old_sr=sample_rate, new_sr=16000)
        hidden = self.encoder(x)

        watermark = self.decoder(hidden)

        if sample_rate != 16000:
            watermark = julius.resample_frac(
                watermark, old_sr=16000, new_sr=sample_rate
            )

        return watermark[..., :length]

    def pad_w_zeros(
        self,
        x: torch.Tensor,
        watermark: torch.Tensor
    ) -> torch.Tensor:
        if not self.future:
            zeros = torch.zeros(x.size(0), x.size(1), x.size(-1) - watermark.size(-1) - 204).cuda()
            actual_watermark = torch.cat([torch.zeros(x.size(0), x.size(1), 204).cuda(), watermark, zeros],
                                         dim=-1) + 1e-9
        else:
            zeros = torch.zeros(x.size(0), x.size(1), x.size(-1) - watermark.size(-1) - (204 + self.future_ts)).cuda()
            # print(torch.zeros(x.size(0), x.size(1), 204).size())   # (3, 1, 204)
            # print(watermark.size())  # (3, 1, 39729)
            # print(zeros.size())  # (3, 1, 17)

            actual_watermark = torch.cat([torch.zeros(x.size(0), x.size(1), (204 + self.future_ts)).cuda(), watermark, zeros],
                                         dim=-1) + 1e-9
        return actual_watermark

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = 16000,
        alpha: float = 1.0
    ) -> tuple[Tensor, Tensor] | None:
        """Apply the watermarking to the audio signal x with a tune-down ratio (defaul 1.0)"""
        if sample_rate is None:
            logger.warning("No sample rate input, setting it to be 16khz")
            sample_rate = 16_000
        list_of_watermark = []

        if int((x.size(-1) - (204+self.future_ts)) / 51) > 0:
            for i in range(int((x.size(-1) - (204+self.future_ts)) / 51)):
                out = self.get_watermark(x[:, :, i*51:204+i*51], sample_rate=sample_rate)
                # i = 0 x[:, :, 0:204]
                # i = 1 x[:, :, 51:204+51]
                list_of_watermark.append(out)

        if len(list_of_watermark) > 0:
            watermark = torch.cat(list_of_watermark, dim=-1)
            actual_watermark = self.pad_w_zeros(x, watermark)
            mask = x != 0
            wm = actual_watermark*mask + 0.0000001
            # wm = self.get_watermark(x, sample_rate=sample_rate)
            return x + alpha * wm, alpha * wm

        else:
            return None

    def training_step(self, batch, batch_idx, wmdetector: torch.nn.Module):
        inputs, labels = batch
        output = wmdetector(self.forward(inputs))

        if output is not None:
            ce_loss = self.CEloss(output, labels)
        else:
            return {"loss": torch.Tensor([0.0])}

        if self.wandb:
            self.log('Cross entropy loss', ce_loss, on_step=True, sync_dist=True)
        return {"loss": ce_loss}


class WatermarkDetector(torch.nn.Module):
    """
    Detect the watermarking from an audio signal
    Args:
        SEANetEncoderKeepDimension (_type_): _description
        nbits (int): The number of bits in the secret message. The result will have size
        of 2 + nbits, where the first two items indicate the possibilities of the audio
        being watermarked (positive / negative scores), the rest is used to decode the secret
        message. In 0bit watermarking (no secret message), the detector just return 2 values.
    """

    def __init__(self, *args, nbits: int = 0, local_detection: bool = True, **kwargs):
        super().__init__()
        encoder = SEANetEncoderKeepDimension(*args, **kwargs)
        last_layer = torch.nn.Conv1d(encoder.output_dim, 2 + nbits, 1)
        self.local_detection = local_detection
        self.detector = torch.nn.Sequential(encoder, last_layer)
        self.nbits = nbits

    def detect_watermark(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = 16000,
        message_threshold: float = 0.5
    ) -> Tuple[float, torch.Tensor]:
        """
        A convenience function that returns a probability of an audio being watermarked,
        together with its message in n_bits (binary) format. If the audio is not watermarked,
        the message will be random.
        Args:
            x: Audio signal, size: batch X frames
            sample_rate: The sample rate of the input audio
            message_threshold: threshold used to convert the watermark output (probability
                of each bits being 0 and 1) into the binary n-bit message
        """
        if sample_rate is None:
            logger.warning("No sample rate input, setting it to be 16khz")
            sample_rate = 16_000
        result, message = self.forward(x, sample_rate=sample_rate)  # b x 2+nbits
        detected = (
            torch.count_nonzero(torch.gt(result[:, 1, :], 0.5)) / result.shape[-1]
        )
        detect_prob = detected.cpu().item()  # type: ignore
        message = torch.gt(message, message_threshold).int()
        return detect_prob, message

    def forward(
        self,
        x: torch.Tensor,
        sample_rate: Optional[int] = 16000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Detect the watermarks from the audio signal
        Args:
            x: Audio signal, size batch x frames
            sample_rate: The sample rate of the input audio
        """
        if sample_rate is None:
            logger.warning("No sample rate input, setting it to be 16khz")
            sample_rate = 16_000
        assert sample_rate
        if sample_rate != 16000:
            x = julius.resample_frac(x, old_sr=sample_rate, new_sr=16000)

        result = self.detector(x)  # b x 2+nbits
        if not self.local_detection:
            result = F.avg_pool1d(result, result.size(-1))  # Apply average pooling on the logits
            result[:, :2] = torch.softmax(result[:, :2, :], dim=1)  # Apply softmax
            return result[:, :2, :], torch.Tensor([0])
        else:
            # hardcode softmax on 2 first units used for detection
            result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)
            return result[:, :2, :], torch.Tensor([0])

