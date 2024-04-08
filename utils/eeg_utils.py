from torchaudio import transforms
from typing import Union
import torch.nn as nn
import einops
import torch
import math

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

    def forward(
            self,
            eegs: torch.Tensor,
        ):
        is_batched = True if len(eegs.shape) == 3 else False
        if not is_batched:
            eegs = einops.rearrange(eegs, "s c -> () s c")
        eegs = torch.tensor(eegs, dtype=torch.float32)
        window_size = min(self.window_size, eegs.shape[1])
        window_stride = min(self.window_stride, window_size // 2)
        mel_fn = transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            f_min=self.min_freq,
            f_max=self.max_freq,
            n_mels=self.mels,
            center=True,
            n_fft=max(128, window_size),
            normalized=True,
            power=1,
            win_length=window_size,
            hop_length=window_stride,
            pad=math.ceil(window_stride//2),
        ).to(self.device).float()
        eegs = einops.rearrange(eegs, "b s c -> b c s")
        spectrogram = mel_fn(eegs)  # (b c m s)
        spectrogram = einops.rearrange(spectrogram, "b c m s -> b s c m")
        if not is_batched:
            spectrogram = einops.rearrange(spectrogram, "b s c m -> (b s) c m")
        return spectrogram