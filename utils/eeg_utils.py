from config import USE_DML, MELS, MELS_WINDOW_SIZE, MELS_WINDOW_STRIDE, MELS_MAX_FREQ, MELS_MIN_FREQ, SAMPLING_RATE
from utils.utils import select_device
from torchaudio import transforms
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import random
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

def apply_augmentation_to_spectrograms(data, time_mask_param, freq_mask_param):
    print("--AUGMENTATION-- apply_regularization flag set to True. Mel spectrograms will be augmented.")
    for i in tqdm(range(len(data)), desc="Applying augmentations to training data..", leave=False):
        mel_spectrogram = data[i]["spectrogram"]
        # Apply SpecAugment if the random number is greater than 0.5
        if random.random() > 0.5:
            mel_spectrogram = spec_augment(mel_spectrogram, time_mask_param, freq_mask_param)
        
        # Apply additive noise if the random number is greater than 0.5
        if random.random() > 0.5:
            mel_spectrogram = add_noise(mel_spectrogram)
        
        # Apply flipping if the random number is greater than 0.75
        if random.random() > 0.75:
            mel_spectrogram = flip(mel_spectrogram)
        data[i]["spectrogram"] = mel_spectrogram
    
    return data

def spec_augment(mel_spectrogram, time_mask_param, freq_mask_param):
    # Apply frequency masking
    freq_masker = transforms.FrequencyMasking(freq_mask_param=freq_mask_param)
    mel_spectrogram = freq_masker(mel_spectrogram)

    # Apply time masking
    time_masker = transforms.TimeMasking(time_mask_param=time_mask_param)
    mel_spectrogram = time_masker(mel_spectrogram)

    return mel_spectrogram

def add_noise(mel_spectrogram):
    noise_level = 0.05  # Additive noise parameters
    noise = torch.randn_like(mel_spectrogram) * noise_level # Generate random noise
    mel_spectrogram += noise # Add noise to the mel spectrogram
    return mel_spectrogram

def flip(mel_spectrogram):
    # Flip along the time axis
    mel_spectrogram = torch.flip(mel_spectrogram, dims=[2])
    return mel_spectrogram