from config import WINDOWS_SIZE, WINDOWS_STRIDE, SAMPLING_RATE, ELECTRODES, NORMALIZE_DATA
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm
import numpy as np
import einops
import mne

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
        
        self.eegs_data, self.labels_data, self.subjects_data = self.load_data()
        print(len(self.eegs_data))
        print(self.eegs_data[0].shape)

        # Normalizes the EEG data if the NORMALIZE_DATA flag is set to True
        if NORMALIZE_DATA:
            print("--NORMALIZATION-- Normalization_data flag set to True. EEG data will be normalized..")
            self.eegs_data = self.normalize_data(self.eegs_data)

        
    def __len__(self):
        # TO DO: Implement the division of the eeg data into windows
        return 0
    
    def __getitem__(self, idx):
        # TO DO: Implement the division of the eeg data into windows
        return "TO DO: Implement __getitem__"

    @abstractmethod
    def load_data(self):
        pass

    def normalize_data(self, eegs_data):
        for i, experiment in enumerate(tqdm(eegs_data, desc="Normalizing EEG data..", unit="experiment", leave=False)):
            print(experiment.shape)
            scaler = mne.decoding.Scaler(info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate, verbose=False, ch_types="eeg"), scalings="mean") # Initializes the scaler
            experiment_scaled = einops.rearrange(experiment, "s c -> () c s")
            print(experiment_scaled.shape)
            experiment_scaled = scaler.fit_transform(experiment_scaled)
            experiment_scaled = einops.rearrange(experiment_scaled, "b c s -> s (b c)") # Normalizes the data
            experiment_scaled = np.nan_to_num(experiment_scaled) # Replaces NaN values with 0
            experiment_scaled = 2 * ((experiment_scaled - experiment_scaled.min(axis=0)) / (experiment_scaled.max(axis=0) - experiment_scaled.min(axis=0))) - 1 # Normalizes between -1 and 1
            eegs_data[i] = experiment_scaled # Updates the EEG data
        return eegs_data
    
    def get_windows(self):
        pass