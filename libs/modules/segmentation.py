import torchaudio
import torch
import torchaudio.transforms as transforms
from torchaudio.functional import band_biquad, highpass_biquad


def apply_k_weighting(signal, sample_rate, treble_cutoff=8000, highpass_cutoff=1000):
    """
    Apply a K-weighting filter which includes a high-pass filter to the signal.
    """
    # Apply the treble filter
    treble_transform = transforms.BiquadFilter(sample_rate, 'treble', f0=treble_cutoff)
    treble_filtered = treble_transform(signal)

    # Apply the high-pass filter
    highpass_transform = transforms.HighpassFilter(sample_rate, highpass_cutoff)
    highpass_filtered = highpass_transform(treble_filtered)

    return highpass_filtered

def divide_signal_into_bands(signal, sample_rate, bands):
    """
    Divides the signal into multiple frequency bands

    :param signal: Input audio (Tensor)
    :param sample_rate: Sample rate of the audio signal (int)
    :param bands: List of tuples (low_freq, high_freq) defining the band boundaries.
    :return: List of signals, each filtered to a specific band.
    """
    filtered_signals = []
    for (central_freq, filter_bandwidth) in bands:
        filtered_signal = band_biquad(signal, sample_rate, central_freq, filter_bandwidth)
        filtered_signals.append(filtered_signal)
    return filtered_signals


if __name__ == "__main__":
    sample_rate = 44100
    duration = 1
    noise = torch.randn((3, sample_rate * duration))

    # central band 20, filter bandwidth 300
    bands = [(20, 300), (300, 3000), (3000, 20000), (20000, 25000)]

    band_signals = divide_signal_into_bands(noise, sample_rate, bands)

    window_size = 1024
    overlap_ratio = 0.2

    segments = segment_signal(band_signals, window_size, overlap_ratio)
    print(f"Number of bands: {len(segments)}")
    print(f"Number of segments: {len(segments[0])}")
    print(f"Each segment shape: {segments[0][0].size()}")


def segment_signal(signal_bands, window_size, overlap_ratio, window_fn=None):
    """
    Segments the signal into windows with specified overlap

    :param signal: Input audio signal (Tensor)
    :param window_size: The size of each window (int)
    :param overlap_ratio: The fraction of window overlapped in each step (float)
    :param window_fn: The window functions to be applied to the window segment (fn)
    :return: Tensor containing windowed segments of the signal.
    """
    # Calculate the step size from the overlap ratio
    step_size = int(window_size * (1 - overlap_ratio))
    segments_list = []
    for signal in signal_bands:
        # Number of segments
        num_segments = (signal.shape[-1] - window_size) // step_size + 1

        # Create an empty tensor to store the segments
        segments = []

        if window_fn is not None:
            # Create the window
            window = window_fn(window_size).to(signal.device)

            for i in range(num_segments):
                start_index = i * step_size
                end_index = start_index + window_size
                segments.append(signal[:, start_index:end_index] * window)
        else:
            for i in range(num_segments):
                start_index = i * step_size
                end_index = start_index + window_size
                segments.append(signal[:, start_index:end_index])

        segments_list.append(segments)

    return segments_list






