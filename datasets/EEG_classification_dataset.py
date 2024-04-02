from config import WINDOWS_SIZE, WINDOWS_STRIDE, SAMPLING_RATE, ELECTRODES, NORMALIZE_DATA, DROP_LAST
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
                 drop_last: bool = DROP_LAST
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
        self.drop_last = drop_last
        self.samples_per_window = int(np.floor(self.sampling_rate * self.window_size)) # Number of samples per window
        self.samples_per_stride = int(np.floor(self.sampling_rate * self.window_stride)) # Number of samples per stride
        
        self.eegs_data, self.labels_data, self.subjects_data = self.load_data()

        # Normalizes the EEG data if the NORMALIZE_DATA flag is set to True
        if NORMALIZE_DATA:
            print("--NORMALIZATION-- Normalization_data flag set to True. EEG data will be normalized..")
            self.eegs_data = self.normalize_data(self.eegs_data)

        # Divide the EEG data into windows
        self.windows = self.get_windows()
        print(self.windows)

        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        # TO DO: Implement the division of the eeg data into windows
        return "TO DO: Implement __getitem__"

    @abstractmethod
    def load_data(self):
        pass

    def normalize_data(self, eegs_data, epsilon=1e-9):
        for i, subject_experiment in enumerate(tqdm(eegs_data, desc="Normalizing EEG data..", unit="experiment", leave=False)):
            scaler = mne.decoding.Scaler(info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate, verbose=False, ch_types="eeg"), scalings="mean") # Initializes the scaler
            for j, trial in enumerate(subject_experiment):
                trial_scaled = einops.rearrange(trial, "s c -> () c s")
                trial_scaled = scaler.fit_transform(trial_scaled)
                trial_scaled = einops.rearrange(trial_scaled, "b c s -> s (b c)") # Normalizes the data
                trial_scaled = np.nan_to_num(trial_scaled) # Replaces NaN values with 0
                #TO DO: Sembra che ci siano esperimenti che valgono sempre 0. Va bene così? Vanno tolti?
                #den = trial_scaled.max(axis=0) - trial_scaled.min(axis=0) + epsilon
                #print(den)
                #if den.any == 0 or den.any() == epsilon:
                #    print(trial_scaled.max(axis=0), trial_scaled.min(axis=0), den)
                trial_scaled = 2 * ((trial_scaled - trial_scaled.min(axis=0)) / (trial_scaled.max(axis=0) - trial_scaled.min(axis=0) + epsilon)) - 1 # Normalizes between -1 and 1
                subject_experiment[j] = trial_scaled # Updates the EEG data
            eegs_data[i] = subject_experiment
        return eegs_data
    
    def get_windows(self):
        windows = []
        for i, subject_experiment in enumerate(tqdm(self.eegs_data, desc="Dividing EEG data into windows..", unit="experiment", leave=False)):
            for j, trial in enumerate(subject_experiment):
                for k in range(0, len(trial), self.samples_per_stride):
                    window = {
                        "experiment": j,
                        "start": k,
                        "end": k + self.samples_per_window,
                        "subject_id": self.subjects_data[i],
                        "labels": self.labels_data[i]
                    }
                    if self.drop_last and (window["end"] - window["start"]) != self.samples_per_window:
                        continue
                    windows.append(window)
        return windows
