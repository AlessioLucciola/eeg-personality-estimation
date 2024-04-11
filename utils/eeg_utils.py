from typing import Union
import torch.nn as nn
import numpy as np
import librosa
import einops
import torch
import math

#TO DO: FUNZIONA SU CPU MA NON SU GPU (DIRECTML)
#Bisogna probabilmente portare su gpu con .to(device) i tensori giusti

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