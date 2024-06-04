from utils.eeg_utils import MergeMels
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
                 num_encoders: int,
                 num_decoders: int,
                 use_encoder_only: bool,
                 merge_mels_typology: str,
                 device: any,
                 positional_encoding: nn.Module = None,
                 use_learnable_token: bool = True
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
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.use_encoder_only = use_encoder_only
        self.positional_encoding = positional_encoding
        self.use_learnable_token = use_learnable_token
        self.merge_mels_typology = merge_mels_typology
        self.device = device
        
        self.dropout = nn.Dropout(p=self.dropout_p) # Dropout layer

        print(self.hidden_size)
        print(self.num_heads)
        print(self.hidden_size % self.num_heads == 0)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                batch_first=True,
                d_model=self.hidden_size,
                dim_feedforward=self.hidden_size*4,
                dropout=self.dropout_p,
                activation=nn.functional.relu,
                nhead=self.num_heads,
            ),
            num_layers=self.num_encoders,
        )

        if not self.use_encoder_only:
            self.labels_embedding = nn.Embedding(len(self.labels), self.hidden_size)
            self.decoder = nn.TransformerDecoder(
                decoder_layer=nn.TransformerDecoderLayer(
                    batch_first=True,
                    d_model=self.hidden_size,
                    dim_feedforward=self.hidden_size*4,
                    dropout=self.dropout_p,
                    activation=nn.functional.relu,
                    nhead=self.num_heads,
                ),
                num_layers=self.num_decoders,
            )

        if self.use_learnable_token:
            self.cls = nn.Embedding(1, self.hidden_size)

        # Prepare the data for the transformer by merging the mel bands
        self.merge_mels = MergeMels(mel_spectrogram=self.mels,
                                    hidden_size=self.hidden_size,
                                    typology=self.merge_mels_typology,
                                    device=self.device
                                    )
        self.merge_mels = self.merge_mels.to(self.device)

        # Add the classifier
        self.fc_layers = []
        self.fc_layers.append(self.dropout)
        self.fc_layers.append(nn.Linear(self.hidden_size, len(self.labels)))
        
        self.classifier = nn.Sequential(*self.fc_layers)
        
    def forward(self, x):
        if not self.use_encoder_only:
            label_indices = torch.arange(len(self.labels), device=x.device).unsqueeze(0) # # Create a 2D tensor of label indices with shape (1, num_labels)
            label_tokens = self.labels_embedding(label_indices)  # Embed the label indices (1, num_labels, hidden_size)
            label_tokens = label_tokens.expand(x.shape[0], -1, -1)  # # Repeat the label tokens across the batch dimension (batch_size, num_labels, hidden_size)
        x = self.merge_mels(x) # Merge the mel bands (merging typology is defined in the configuration file)
        #print(x.shape)
        if self.positional_encoding is not None:
            x = x + self.positional_encoding(x) # Add positional encoding
            if not self.use_encoder_only:
                label_tokens = label_tokens + self.positional_encoding(label_tokens) # Add positional encoding to the label tokens
        #print(self.cls.weight.shape)
        if self.use_learnable_token:
            cls_token = self.cls(torch.tensor([0], device=self.device)).repeat(x.shape[0], 1, 1)  # Expand cls token to the batch size
            x = torch.cat([cls_token, x], dim=1)  # Add learnable token
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        if self.use_encoder_only:
            x = x[:, 0, :]
        else:
            x = self.decoder(label_tokens, x)[:, 0, :]
        #print(x.shape)
        x = self.classifier(x)
        #print(x.shape)
        return x