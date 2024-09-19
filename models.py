import logging
from urllib.parse import urlparse  # noqa: F401
from typing import (  # type: ignore[attr-defined]
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)
import julius
import torch

from libs.modules.seanet import *

logger = logging.getLogger("VoiceWatermark")


class MsgProcessor(torch.nn.Module):
    """
    Apply the secret message to the encoder output.
    Args:
        nbits: Number of bits used to generate the message. Must be non-zero
        hidden_size: Dimension of the encoder output.
    """
    def __init__(self, nbits: int, hidden_size: int):
        super().__init__()
        assert nbits > 0, "MsgProcessor should not be built in 0bit watermarking"
        self.nbits = nbits
        self.hidden_size = hidden_size
        self.msg_processor = torch.nn.Embedding(2 * nbits, hidden_size)

    def forward(self, hidden: torch.Tensor, msg: torch.Tensor) -> torch.Tensor:
        """
        Build the embedding map: 2 x k -> k x h, them sum on the first dim
        Args:
            hidden: The encoder output, size: batch x hidden x frames
            msg: The secret message, size: batch x k
        """
        # Create indices to take from embedding layer
        indices = 2 * torch.arange(msg.shape[-1]).to(msg.device)  # k: 0 2 4 ... 2k
        indices = indices.repeat(msg.shape[0], 1)  # b x k
        indices = (indices + msg).long()
        msg_aux = self.msg_processor(indices)  # b x k -> b x k x h
        msg_aux = msg_aux.sum(dim=-2)  # b x k x h -> b x h
        msg_aux = msg_aux.unsqueeze(-1).repeat(
            1, 1, hidden.shape[2]
        )  # b x h -> b x h x t/f
        hidden = hidden + msg_aux  # -> b x h x t/f
        return hidden


class WatermarkModel(torch.nn.Module):
    """
    Generate watermarking for a given audio signal
    """

    def __init__(
            self,
            encoder: torch.nn.Module,
            decoder: torch.nn.Module,
            model_config,
            n_fft: int = 320,  # 512,
            hop_length: int = np.prod(list(reversed([8, 5, 4]))),
            future_ts: float = 50,
            future: bool = True,
            power: float = 0.008,
            wandb: bool = False,
            msg_processor: Optional[torch.nn.Module] = None
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

        self.path_model = model_config["test"]["model_path"]
        self.model_name = model_config["test"]["model_name"]
        self.index = model_config["test"]

        self.msg_processor = msg_processor
        self._message: Optional[torch.Tensor] = None

    @property
    def message(self) -> Optional[torch.Tensor]:
        return self._message

    @message.setter
    def message(self, message: torch.Tensor) -> None:
        self._message = message

    def get_watermark(
            self,
            x: torch.Tensor,
            sample_rate: Optional[int] = 16000,
            message: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Get the watermark from an audio tensor.
        Args:
            x: Audio signal, size: batch x frames
            sample_rate: The sample rate of input audio (default 16khz as
                currently supported by the main model)
            message: An optional binary message, size: batch x k
        """
        length = x.size(-1)
        if sample_rate is None:
            logger.warning("No sample rate input, setting it to be 16khz")
            sample_rate = 16_000
        assert sample_rate
        if sample_rate != 16000:
            x = julius.resample_frac(x, old_sr=sample_rate, new_sr=16000)
        hidden = self.encoder(x)
        hidden = hidden.reshape(x.shape[0], 1, 512)

        if self.msg_processor is not None:
            if message is None:
                if self.message is None:
                    message = torch.randint(
                        0, 2, (x.shape[0], self.msg_processor.nbits), device=x.device
                    )
                else:
                    message = self.message.to(device=x.device)
            else:
                message = message.to(device=x.device)
            hidden = self.msg_processor(hidden, message)

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
            message: Optional[torch.Tensor] = None,
            alpha: float = 1.0
    ):
        """Apply the watermarking to the audio signal x with a tune-down ratio (defaul 1.0)"""
        if sample_rate is None:
            logger.warning("No sample rate input, setting it to be 16khz")
            sample_rate = 16_000

        x_spect = self.stft(x).permute(0, 3, 1, 2)  # [b, freq_bins, time_frames, 2] -> [b, 2, freq_bins, time_frames]
        list_of_watermark = []

        if int((x_spect.size(-1) - (204 + self.future_ts)) / 51) > 0:
            for i in range(int((x_spect.size(-1) - (204 + self.future_ts)) / 51)):
                out = self.get_watermark(x_spect[:, :, :, i * 51:204 + i * 51], sample_rate=sample_rate,
                                         message=message)
                list_of_watermark.append(out)
                # 2.05s has 2.05*16000 = 32800 samples
                # n_fft (frame_size) = 320dis
                # hop_length = 160
                # frames = (32800-320)/160 + 1 = 204 frames
                # freq_bins = n_fft (frame_size) // 2 + 1 = 161
        if len(list_of_watermark) > 0:
            watermark_wav = torch.cat(list_of_watermark, dim=2)[:, 0, :]  # squeeze out the extra 1 dimension
            all_watermark_wav = self.power * torch.max(torch.abs(x), dim=1)[0].unsqueeze(1) * watermark_wav
            actual_watermark_wav = self.pad_w_zeros(x, all_watermark_wav)
            mask = x != 0
            wm = actual_watermark_wav * mask + 0.0000001
            return x + alpha * wm, alpha * wm

        else:
            print("None type warning!")
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
        last_layer = nn.Conv1d(1, 2, 1)
        self.detector = nn.Sequential(detector, last_layer)

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

    def decode_message(self, result: torch.Tensor) -> torch.Tensor:
        """
        Decode the message from the watermark result (batch x nbits x frames)
        Args:
            result: watermark result (batch x nbits x frames)
        Returns:
            The message of size batch x nbits, indicating probability of 1 for each bit
        """
        assert (result.dim() > 2 and result.shape[1] == self.nbits) or (
            result.dim() == 2 and result.shape[0] == self.nbits
        ), f"Expect message of size [,{self.nbits}, frames] (get {result.size()})"
        decoded_message = result.mean(dim=-1)
        return torch.sigmoid(decoded_message)

    def stft(self, x):
        window = torch.hann_window(self.n_fft).to(x.device)
        tmp = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=True)
        tmp = torch.view_as_real(tmp)
        return tmp

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
        orig_length = x.shape[-1]
        x_spect = self.stft(x).permute(0, 3, 1, 2)
        result = self.detector(x_spect)[..., :orig_length]  # b x 2+nbits x length
        result[:, :2, :] = torch.softmax(result[:, :2, :], dim=1)  # Apply softmax
        # message = self.decode_message(result[:, 2:, :])  # Decode the message
        return result[:, :2, :], torch.tensor([0])
