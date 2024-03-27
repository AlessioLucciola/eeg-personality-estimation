from config import WINDOWS_SIZE, WINDOWS_STRIDE, SAMPLING_RATE, ELECTRODES
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import List
import os

class EEGClassificationDataset(Dataset, ABC):
    def __init__(self,
                 data_path: str,
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
        self.eeg_data, self.labels_data, self.subject_ids_data = self.load_data()
        
    def __len__(self):
        # TO DO: Implement the division of the eeg data into windows
        return "TO DO: Implement __len__"
    
    def __getitem__(self, idx):
        # TO DO: Implement the division of the eeg data into windows
        return "TO DO: Implement __getitem__"
    
    @staticmethod
    @abstractmethod
    def get_subject_ids_static(path: str):
        pass

    @abstractmethod
    def load_data(self):
        pass
    
if __name__ == "__main__":
    dataset = EEGClassificationDataset()