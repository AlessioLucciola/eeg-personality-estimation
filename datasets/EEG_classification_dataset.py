from config import WINDOWS_SIZE, WINDOWS_STRIDE, SAMPLING_RATE, ELECTRODES
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import List
import os

class EEGClassificationDataset(Dataset, ABC):
    def __init__(self,
                 data_path: str,
                 metadata_path: str,
                 dataset_name: str,
                 subject_ids: List[str],
                 labels: List[str],
                 labels_classes: int,
                 sampling_rate: int = SAMPLING_RATE,
                 electrodes: List[str] = ELECTRODES,
                 window_size: int = WINDOWS_SIZE,
                 window_stride: int = WINDOWS_STRIDE,
                ):
        super().__init__()

        # TO DO: Check parameters validity
        self.data_path = data_path
        self.metadata_path = metadata_path
        self.dataset_name = dataset_name
        self.subject_ids = subject_ids
        self.labels = labels
        self.labels_classes = labels_classes
        self.sampling_rate = sampling_rate
        self.electrodes = electrodes
        self.window_size = window_size
        self.window_stride = window_stride
        
        #self.eeg_data, self.labels_data, self.subject_ids_data = self.load_data()
        
    def __len__(self):
        # TO DO: Implement the division of the eeg data into windows
        return 0
    
    def __getitem__(self, idx):
        # TO DO: Implement the division of the eeg data into windows
        return "TO DO: Implement __getitem__"

    @abstractmethod
    def load_data(self):
        pass
    
#if __name__ == "__main__":
#    dataset = EEGClassificationDataset()