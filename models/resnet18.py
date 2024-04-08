import pytorch_lightning as pl
from typing import List

class ResNet18(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 sampling_rate: int,
                 labels: List[str],
                 labels_classes: int,
                 mels: int,
                 mel_window_size: float,
                 mel_window_stride: float,
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
        self.device = device