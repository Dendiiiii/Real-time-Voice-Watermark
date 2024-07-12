import logging
from typing import Optional, Tuple
from torch import Tensor
import julius
from libs.modules.seanet import *

logger = logging.getLogger("VoiceWatermark")


class WatermarkModel(torch.nn.Module):
    """
    Generate watermarking for a given audio signal
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        n_fft: int = 320,  # 512,
        hop_length: int = np.prod(list(reversed([8, 5, 4]))),
        future_ts: float = 50,
        future: bool = True,
        power: float = 0.008,
        wandb: bool = False
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.CEloss = torch.nn.CrossEntropyLoss()
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

        hidden = self.encoder(x).reshape(x.shape[0], 1, 512)

        watermark = self.decoder(hidden)

        if sample_rate != 16000:
            watermark = julius.resample_frac(
                watermark, old_sr=16000, new_sr=sample_rate
            )

        return watermark[..., :length]

    def pad_w_zeros(
        self,
        x: torch.Tensor,
        watermark_wav: torch.Tensor
    ) -> torch.Tensor:
        if not self.future:
            zeros = torch.zeros(x.size(0), x.size(1) - watermark_wav.size(1) - 204).to(x.device)
            actual_watermark_wav = torch.cat([torch.zeros(x.size(0), 204).to(x.device), watermark_wav, zeros],
                                             dim=1) + 1e-9
        else:
            zeros = torch.zeros(x.size(0), x.size(1) - watermark_wav.size(1) - (204 + self.future_ts)).to(x.device)
            actual_watermark_wav = torch.cat(
                [torch.zeros(x.size(0), (204 + self.future_ts)).to(x.device), watermark_wav, zeros],
                dim=1) + 1e-9
        return actual_watermark_wav

    def stft(self, x):
        window = torch.hann_window(self.n_fft).to(x.device)
        tmp = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        tmp = torch.view_as_real(tmp)
        return tmp

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

        x_spect = self.stft(x).permute(0, 3, 1, 2)  # [b, freq_bins, time_frames, 2] -> [b, 2, freq_bins, time_frames]

        list_of_watermark = []

        if int((x_spect.size(-1) - (204+self.future_ts)) / 51) > 0:
            for i in range(int((x_spect.size(-1) - (204+self.future_ts)) / 51)):
                out = self.get_watermark(x_spect[:, :, :, i*51:204+i*51], sample_rate=sample_rate)
                list_of_watermark.append(out)

        if len(list_of_watermark) > 0:
            watermark_wav = torch.cat(list_of_watermark, dim=2)[:, 0, :]  # squzze out the extra 1 dimension
            all_watermark_wav = self.power*torch.max(torch.abs(x), dim=1)[0].unsqueeze(1) * watermark_wav
            actual_watermark_wav = self.pad_w_zeros(x, all_watermark_wav)
            mask = x != 0
            wm = actual_watermark_wav*mask + 0.0000001
            return x + alpha * wm, alpha * wm

        else:
            return None


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

    def __init__(self, *args, nbits: int = 0, detector: torch.nn.Module, n_fft: int = 320,
                 hop_length: int = np.prod(list(reversed([8, 5, 4]))), **kwargs):
        super().__init__()

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.nbits = nbits
        last_layer = nn.Conv1d(2, 2, 1)
        self.detector = torch.nn.Sequential(detector, last_layer)

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
            torch.count_nonzero(torch.gt(result[:, :], 0.5)) / result.shape[-1]
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

        result = self.detector(x)  # b x 2+nbits x length
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)  # Apply softmax
        return result[:, :2, :], torch.Tensor([0])

