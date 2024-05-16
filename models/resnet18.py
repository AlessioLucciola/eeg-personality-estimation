from utils.train_utils import add_dropout_to_model
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
                 add_dropout_to_resnet: bool,
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
        self.add_dropout_to_resnet = add_dropout_to_resnet
        self.device = device
        
        self.dropout = nn.Dropout(p=self.dropout_p) # Dropout layer
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if self.pretrained else None) # Load pretrained weights if pretrained
        self.resnet18.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) # Modify the first convolutional layer to accept the EEG data

        # Add the classifier
        self.fc_layers = []
        self.fc_layers.append(self.dropout)
        self.fc_layers.append(nn.Linear(512, len(self.labels)))
        
        self.classifier = nn.Sequential(*self.fc_layers)
        self.resnet18.fc = self.classifier

        # Freeze all layers except the first convolutional layer and the fc layer if pretrained
        if self.pretrained:
            for name, param in self.resnet18.named_parameters():
                if name not in ['conv1.weight', 'fc.weight', 'fc.bias']:
                    param.requires_grad = False

        if self.add_dropout_to_resnet:
            self.resnet18 = add_dropout_to_model(self.resnet18, self.dropout_p)
        
    def forward(self, x):
        x = self.resnet18(x) # Forward pass
        return x