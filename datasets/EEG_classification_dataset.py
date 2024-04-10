from config import WINDOWS_SIZE, WINDOWS_STRIDE, SAMPLING_RATE, ELECTRODES, NORMALIZE_DATA, DROP_LAST, PRINT_DATASET_DEBUG, MAKE_PLOTS
from plots.plots import plot_amplitudes_distribution, plot_labels_distribution, plot_sample, plot_subjects_distribution
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

        # Discard corrupted experiments
        self.eegs_data = self.discard_corrupted_experiments(self.eegs_data)

        # Plot EEG data sample before normalization
        #if MAKE_PLOTS:
        #    for i in tqdm(range(len(self.eegs_data)), desc="--PLOTS-- Plotting EEG data sample before normalization..", unit="subject", leave=False):
        #        plot_sample(self.eegs_data[i][0], self.electrodes, dataset_name=self.dataset_name, data_normalized=False, title=f"EEG data sample for subject {self.subjects_data[i]} before normalization")

        # Normalizes the EEG data if the NORMALIZE_DATA flag is set to True
        if NORMALIZE_DATA:
            print("--NORMALIZATION-- Normalization_data flag set to True. EEG data will be normalized..")
            self.eegs_data = self.normalize_data(self.eegs_data)
        
        # Plot the EEG data sample after normalization
        #if MAKE_PLOTS:
        #    for i in tqdm(range(len(self.eegs_data)), desc="--PLOTS-- Plotting EEG data sample after normalization..", unit="subject", leave=False):
        #        plot_sample(self.eegs_data[i][0], self.electrodes, dataset_name=self.dataset_name, data_normalized=True, title=f"EEG data sample for subject {self.subjects_data[i]} after normalization")

        # Plot the amplitudes distribution of the EEG data after normalization
        if MAKE_PLOTS:
            print("--PLOTS-- Plotting amplitudes distribution of EEG data after normalization..")
            all_eegs = np.concatenate([np.concatenate(exp) for exp in self.eegs_data])
            plot_amplitudes_distribution(all_eegs, self.electrodes, dataset_name=self.dataset_name, title=f"Amplitudes distribution of EEG data after normalization")
            del all_eegs

        # Plot the subjects distribution
        if MAKE_PLOTS:
            print("--PLOTS-- Plotting subjects distribution..")
            subject_samples_num = {}
            for i, s_id in enumerate(self.subject_ids):
                subject_samples_num[s_id] = len(self.eegs_data[i])
            #plot_subjects_distribution(subject_samples_num, dataset_name=self.dataset_name, title="Subjects distribution")
            del subject_samples_num

        # Plot the labels distribution
        if MAKE_PLOTS:
            print("--PLOTS-- Plotting labels distribution..")
            labels_num = {label: 0 for label in range(len(self.labels))}
            for i, subject_labels in enumerate(self.labels_data):
                for label in subject_labels.keys():
                    if subject_labels[label] == 1:
                        labels_num[label] += 1
            plot_labels_distribution(self.labels, labels_num, discretized_labels=True, dataset_name=self.dataset_name, title="Distribution of labels")
            del labels_num

        # Divide the EEG data into windows
        self.windows = self.get_windows()
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window = self.windows[idx]
        return {
            "eeg_data": window["eeg_data"].astype(np.float32),
            "sample_rate": self.sampling_rate,
            "subject_id": window["subject_id"],
            "labels": window["labels"],
        }

    @abstractmethod
    def load_data(self):
        pass

    def discard_corrupted_experiments(self, eegs_data):
        data_discarded = False
        for i, subject_experiment in enumerate(tqdm(eegs_data, desc="--DATASET--Discarding corrupted experiments..", unit="experiment", leave=False)):
            eegs_data[i] = [trial for trial in subject_experiment if np.count_nonzero(np.isnan(trial)) <= trial.size*0.9]
            discarded_experiments = len(subject_experiment) - len(eegs_data[i])
            if discarded_experiments > 0:
                data_discarded = True
                if PRINT_DATASET_DEBUG:
                    print(f"Subject {i} - Discarded {discarded_experiments} corrupted experiments")
        if data_discarded:
            print("WARNING: Some experiments were discarded due to corruption or null values.")
        else :
            print("No corrupted experiments found.")
        return eegs_data

    def normalize_data(self, eegs_data, epsilon=1e-9):
        for i, subject_experiment in enumerate(tqdm(eegs_data, desc="Normalizing EEG data..", unit="experiment", leave=False)):
            scaler = mne.decoding.Scaler(info=mne.create_info(ch_names=self.electrodes, sfreq=self.sampling_rate, verbose=False, ch_types="eeg"), scalings="mean") # Initializes the scaler
            for j, trial in enumerate(subject_experiment):
                trial_scaled = einops.rearrange(trial, "s c -> () c s")
                trial_scaled = scaler.fit_transform(trial_scaled)
                trial_scaled = einops.rearrange(trial_scaled, "b c s -> s (b c)") # Normalizes the data
                trial_scaled = np.nan_to_num(trial_scaled) # Replaces NaN values with 0
                trial_scaled = 2 * ((trial_scaled - trial_scaled.min(axis=0)) / (trial_scaled.max(axis=0) - trial_scaled.min(axis=0) + epsilon)) - 1 # Normalizes between -1 and 1
                subject_experiment[j] = trial_scaled # Updates the EEG data
            eegs_data[i] = subject_experiment
        return eegs_data
    
    def get_windows(self):
        windows = []
        for i, subject_experiment in enumerate(tqdm(self.eegs_data, desc="Dividing EEG data into windows..", unit="experiment", leave=False)):
            for j, trial in enumerate(subject_experiment):
                for k in range(0, len(trial), self.samples_per_stride):
                    window_start = k # Start of the window
                    window_end = k + self.samples_per_window # End of the window
                    subject_id = self.subjects_data[i]
                    if window_end > len(trial): # If the window is larger than actual the EEG data
                        window_end = len(trial) # Adjusts the window size
                    window_eeg = trial[window_start:window_end]
                    if window_eeg.shape[0] < self.samples_per_window:
                        if self.drop_last:
                            if PRINT_DATASET_DEBUG:
                                print(f"WARNING: Window shape of subject {subject_id} experiment {j} is {window_eeg.shape[0]} instead of {self.samples_per_window}. Window will be discarded since drop_last flag is True.")
                            continue
                        else:
                            if PRINT_DATASET_DEBUG:
                                print(f"WARNING: Window shape of subject {subject_id} experiment {j} is {window_eeg.shape[0]} instead of {self.samples_per_window}. Data will be zero-padded.")
                            window_eeg = np.concatenate((window_eeg, np.zeros((self.samples_per_window - window_eeg.shape[0], window_eeg.shape[1]))), axis=0) # Zero-pads the data
                    window = {
                        "experiment": j,
                        "start": window_start,
                        "end": window_end,
                        "eeg_data": window_eeg,
                        "subject_id": self.subjects_data[i],
                        "labels": self.labels_data[i]
                    }
                    windows.append(window)
        return windows
