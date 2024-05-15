from einops.layers.torch import Rearrange
from typing import List
from torch import nn
import torch

class ViT(nn.Module):
    def __init__(self,
                 in_channels: int,
                 labels: List[str],
                 labels_classes: int,
                 dropout_p: float,
                 mels: int,
                 hidden_size: int,
                 num_heads: int,
                 num_layers: int,
                 device: any
                ):
        super().__init__()
        
        # TO DO: Check parameters validity
        self.in_channels = in_channels
        self.labels = labels
        self.labels_classes = labels_classes
        self.dropout_p = dropout_p
        self.mels = mels
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        
        self.dropout = nn.Dropout(p=self.dropout_p) # Dropout layer

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                batch_first=True,
                d_model=self.hidden_size,
                dim_feedforward=self.hidden_size * 4,
                dropout=self.dropout_p,
                activation=nn.functional.selu,
                nhead=self.num_heads,
            ),
            num_layers=num_layers,
        )

        # TO DO: Add the decoder layer and the associated logic to deal with it

        # Prepare the data for the transformer by merging the mel bands
        self.merge_mels = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_size,
                kernel_size=(self.mels, 1),
                stride=1,
                padding=0,
            ),
            Rearrange("b c m s -> b s (c m)")
        )

        # TO DO: Add positional encoding

        # Add the classifier
        self.fc_layers = []
        self.fc_layers.append(self.dropout)
        self.fc_layers.append(nn.Linear(self.hidden_size, len(self.labels)))
        
        self.classifier = nn.Sequential(*self.fc_layers)
        
    def forward(self, x):
        print(x.shape)
        x = self.merge_mels(x) # Merge the mel bands (b c s m -> b s (c m))
        print(x.shape)
        x = self.encoder(x)
        print(x.shape)
        x = x.mean(dim=1) # Average the output of the transformer
        print(x.shape)
        x = self.classifier(x)
        print(x.shape)
        return torch.sigmoid(x)