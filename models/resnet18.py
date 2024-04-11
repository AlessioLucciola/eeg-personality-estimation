from utils.eeg_utils import MelSpectrogram
import torchvision.models as models
from typing import List
from torch import nn
import torch

class ResNet18(nn.Module):
    def __init__(self,
                 in_channels: int,
                 sampling_rate: int,
                 labels: List[str],
                 labels_classes: int,
                 mels: int,
                 mel_window_size: float,
                 mel_window_stride: float,
                 mel_min_freq: float,
                 mel_max_freq: float,
                 dropout_p: float,
                 learning_rate: float,
                 weight_decay: float,
                 device: any
                 ):
        super().__init__()
        
        # TO DO: Check parameters validity
        self.in_channels = in_channels
        self.sampling_rate = sampling_rate
        self.labels = labels
        self.labels_classes = labels_classes
        self.mels = mels
        self.mel_window_size = mel_window_size
        self.mel_window_stride = mel_window_stride
        self.dropout_p = dropout_p
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.mel_min_freq = mel_min_freq
        self.mel_max_freq = mel_max_freq
        self.device = device
        self.spectrogram_module = MelSpectrogram(sampling_rate=self.sampling_rate,
                                            window_size=self.mel_window_size,
                                            window_stride=self.mel_window_stride,
                                            device=self.device,
                                            mels=self.mels,
                                            min_freq=self.mel_min_freq,
                                            max_freq=self.mel_max_freq
                                         )
        
        self.resnet18 = models.resnet18(weights=None)
        self.resnet18.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet18.fc = nn.Linear(512, len(self.labels))
    def forward(self, eegs):
        eegs = eegs.to(self.device)
        spectrogram = self.spectrogram_module(eegs).to(self.device)
        x = self.resnet18(spectrogram)
        del eegs # Free memory to avoid DirectML errors
        del spectrogram # Free memory to avoid DirectML errors
        return torch.sigmoid(x)