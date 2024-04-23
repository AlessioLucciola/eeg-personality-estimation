from utils.eeg_utils import MelSpectrogram
from typing import List
from torch import nn
import torch

class CustomCNN(nn.Module):
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
                 weight_decay: float,
                 device: any
                ):
        super(CustomCNN, self).__init__()
        self.in_channels = in_channels
        self.sampling_rate = sampling_rate
        self.labels = labels
        self.labels_classes = labels_classes
        self.mels = mels
        self.mel_window_size = mel_window_size
        self.mel_window_stride = mel_window_stride
        self.dropout_p = dropout_p
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

        # Define the layers of your CNN
        self.dropout = nn.Dropout(p=self.dropout_p) # Dropout layer
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, len(self.labels))

    def forward(self, eegs):
        eegs = eegs.to(self.device) # Move data to device
        spectrogram = self.spectrogram_module(eegs).to(self.device) # Compute the mel spectrogram

        # Apply convolutional layers with ReLU activation and dropout
        x = self.relu(self.conv1(spectrogram))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout(x)
        
        # Flatten the output for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Apply fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        del eegs # Free memory to avoid DirectML errors
        del spectrogram # Free memory to avoid DirectML errors
        
        return x