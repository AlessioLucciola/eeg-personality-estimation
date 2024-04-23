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
                 pretrained: bool,
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
        self.pretrained = pretrained
        self.device = device
        self.spectrogram_module = MelSpectrogram(sampling_rate=self.sampling_rate,
                                            window_size=self.mel_window_size,
                                            window_stride=self.mel_window_stride,
                                            device=self.device,
                                            mels=self.mels,
                                            min_freq=self.mel_min_freq,
                                            max_freq=self.mel_max_freq
                                        )
        
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if self.pretrained else None) # Load pretrained weights if pretrained
        self.resnet18.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Modify the first convolutional layer to accept the EEG data
        self.resnet18.fc = nn.Linear(512, len(self.labels)) # Modify the fully connected layer to output the number of labels

        # Freeze all layers except the first convolutional layer and the fc layer if pretrained
        if self.pretrained:
            for name, param in self.resnet18.named_parameters():
                if name not in ['conv1.weight', 'fc.weight', 'fc.bias']:
                    param.requires_grad = False
        
    def forward(self, eegs):
        eegs = eegs.to(self.device) # Move data to device
        spectrogram = self.spectrogram_module(eegs).to(self.device) # Compute the mel spectrogram
        x = self.resnet18(spectrogram) # Forward pass
        del eegs # Free memory to avoid DirectML errors
        del spectrogram # Free memory to avoid DirectML errors
        return torch.sigmoid(x)