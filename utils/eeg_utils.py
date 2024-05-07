from config import USE_DML, MELS, MELS_WINDOW_SIZE, MELS_WINDOW_STRIDE, MELS_MAX_FREQ, MELS_MIN_FREQ, SAMPLING_RATE
from utils.utils import select_device
from torchaudio import transforms
import torch.nn as nn
import numpy as np
import einops
import torch
import math

class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sampling_rate=SAMPLING_RATE,
        window_size=MELS_WINDOW_SIZE,
        window_stride=MELS_WINDOW_STRIDE,
        mels=MELS,
        min_freq=MELS_MIN_FREQ,
        max_freq=MELS_MAX_FREQ,
        device=select_device(),
        get_decibels=False
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.device = device
        self.mels = mels
        self.window_size = math.floor(window_size * self.sampling_rate)
        self.window_stride = math.floor(window_stride * self.sampling_rate)
        self.get_decibels = get_decibels

    def forward(self, eeg_data):
        # Check if the input is batched (b, s, c)
        is_batched = True if len(eeg_data.shape) == 3 else False
        # If not batched, add a batch dimension
        if not is_batched:
            eeg_data = einops.rearrange(eeg_data, "s c -> () s c")
        # Convert the numpy array to a torch tensor if needed
        if isinstance(eeg_data, np.ndarray):
            eeg_data = torch.from_numpy(eeg_data).float()

        # Initialize the mel spectrogram function
        mel_fn = transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            f_min=self.min_freq,
            f_max=self.max_freq,
            n_mels=self.mels,
            center=True,
            n_fft=max(128, self.window_size),
            normalized=True,
            power=1,
            win_length=self.window_size,
            hop_length=self.window_stride,
            pad_mode="constant",
        ).float()
        
        # DirectML (AMD GPU) does not support torchaudio mel spectrogram computation. Hence, move the data to the CPU.
        if USE_DML:
            eeg_data = eeg_data.cpu()
            mel_fn = mel_fn.cpu()
    
        eeg_data = einops.rearrange(eeg_data, "b s c -> b c s") # Switch to (b, c, s)
        spectrogram = mel_fn(eeg_data)  # Compute mel spectrogram of the eeg_data - (b c m s)
        if self.get_decibels:
            spectrogram = 10 * torch.log10(spectrogram) # Convert to dB scale
        spectrogram = einops.rearrange(spectrogram, "b s c m -> b c s m") # Switch back to (b, c, s, m)
        # Rearrange the dimensions in case the input was not batched (useful for plotting)
        if not is_batched:
            spectrogram = einops.rearrange(spectrogram, "b c s m -> (b s) c m")
        return spectrogram

'''
class MelSpectrogram(nn.Module):
    def __init__(
            self,
            sampling_rate: int,
            window_size: Union[int, float],
            window_stride: Union[int, float],
            device: any,
            mels: int = 8,
            min_freq: int = 0,
            max_freq: int = 50,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.device = device
        self.mels = mels
        self.window_size = math.floor(window_size * self.sampling_rate)
        self.window_stride = math.floor(window_stride * self.sampling_rate)

    def forward(self, eegs):
        is_batched = True if len(eegs.shape) == 3 else False
        if not is_batched:
            eegs = einops.rearrange(eegs, "s c -> () s c")
        window_size = min(self.window_size, eegs.shape[1])
        window_stride = min(self.window_stride, window_size // 2)
        
        spectrograms = []
        eegs = einops.rearrange(eegs, "b s c -> b c s")
        for eeg in eegs:
            mel_spec = self.compute_mel_spectrogram(eeg.cpu().numpy(), window_size=window_size, window_stride=window_stride) # DirectML (GPU) does not support the mel spectrogram computation
            spectrograms.append(mel_spec)
        spectrograms = np.stack(spectrograms)
        spectrograms = torch.tensor(spectrograms, dtype=torch.float32)
        return spectrograms

    def compute_mel_spectrogram(self, eeg, window_size, window_stride):
        mel_spec = librosa.feature.melspectrogram(
            y=eeg,
            sr=self.sampling_rate,
            n_fft=max(128, self.window_size),
            hop_length=window_stride,
            win_length=window_size,
            power=1,
            n_mels=self.mels,
            fmin=self.min_freq,
            fmax=self.max_freq,
            pad_mode="constant",
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=1)  # Convert to dB scale
        return mel_spec_db
'''