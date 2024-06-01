from config import RANDOM_SEED, BATCH_SIZE, VALIDATION_SCHEME, KFOLDCV, SPLIT_RATIO, APPLY_AUGMENTATION, AUGMENTATION_FREQ_MAX_PARAM, AUGMENTATION_TIME_MAX_PARAM, AUGMENTATION_METHODS, SUBJECTS_LIMIT
from datasets.EEG_classification_dataset import EEGClassificationDataset
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from typing import Optional, List
import random
import torch

# TO DO: Remove from here
from config import AMIGOS_FILES_DIR, AMIGOS_METADATA_FILE
from datasets.AMIGOS_dataset import AMIGOSDataset
from utils.eeg_utils import apply_augmentation_to_spectrograms

class EEG_dataloader(DataLoader):
    def __init__(self,
                 dataset: EEGClassificationDataset,
                 seed: int = RANDOM_SEED,
                 batch_size: int = BATCH_SIZE,
                 validation_scheme: str = VALIDATION_SCHEME,
                 k_folds: Optional[int] = KFOLDCV,
                 split_ratio: Optional[float] = SPLIT_RATIO,
                 subjects_limit: Optional[int] = SUBJECTS_LIMIT,
                 apply_augmentation: bool = APPLY_AUGMENTATION,
                 augmentation_methods: List[str] = AUGMENTATION_METHODS,
                 augmentation_freq_max_param: float = AUGMENTATION_FREQ_MAX_PARAM,
                 augmentation_time_max_param: float = AUGMENTATION_TIME_MAX_PARAM
                ):
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.validation_scheme = validation_scheme
        self.k_folds = k_folds
        self.split_ratio = split_ratio
        self.subjects_limit = subjects_limit
        self.apply_augmentation = apply_augmentation
        self.augmentation_methods = augmentation_methods
        if self.apply_augmentation:
            print("--AUGMENTATION-- apply_regularization flag set to True. Mel spectrograms will be augmented.")
            self.augmentation_freq_max_param = augmentation_freq_max_param
            self.augmentation_time_max_param = augmentation_time_max_param

        if self.validation_scheme == "LOOCV":
            self.dataloaders = self.loocv()
        elif self.validation_scheme == "K-FOLDCV":
            self.dataloaders = self.kfold()
        elif self.validation_scheme == "SPLIT":
            self.dataloaders = self.split()
        else:
            raise ValueError(f"Validation scheme {self.validation_scheme} is not supported.")
    
    # Standard train-test split
    def split(self):
        print(f"--VALIDATION SCHEME-- Split is selected with test_size of {self.split_ratio*100}%.")
        train_df, test_df = train_test_split(self.dataset, test_size=self.split_ratio, random_state=self.seed)
        if self.apply_augmentation:
            train_df = apply_augmentation_to_spectrograms(train_df, aug_to_apply=self.augmentation_methods, time_mask_param=self.augmentation_time_max_param, freq_mask_param=self.augmentation_freq_max_param, k_fold_index=None)
        train_dataloader = DataLoader(train_df, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_df, batch_size=self.batch_size, shuffle=False)
        return tuple((train_dataloader, test_dataloader))
    
    # Leave-One-Out Cross Validation (LOOCV) based on subject_id
    def loocv(self):
        print("--VALIDATION SCHEME-- Leave-One-Out Cross Validation (LOOCV) is selected.")
        loo_dataloaders = {}
        if self.subjects_limit is not None:
            limit = min(self.subjects_limit, len(self.dataset.subject_ids))
            subjects_list = self.dataset.subject_ids[:limit]
        else:
            subjects_list = self.dataset.subject_ids
        for i, subject_id in enumerate(subjects_list):
            train_idx = [j for j in range(len(self.dataset.windows)) if self.dataset.windows[j]['subject_id'] != subject_id]
            val_idx = [j for j in range(len(self.dataset.windows)) if self.dataset.windows[j]['subject_id'] == subject_id]
            train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
            val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
            if self.apply_augmentation:
                train_dataset = apply_augmentation_to_spectrograms(train_dataset, aug_to_apply=self.augmentation_methods, time_mask_param=self.augmentation_time_max_param, freq_mask_param=self.augmentation_freq_max_param, k_fold_index=i+1)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            loo_dataloaders[tuple((i, subject_id))] = tuple((train_dataloader, val_dataloader))
        return loo_dataloaders
    
    # K-Fold Cross Validation
    def kfold(self):
        print(f"--VALIDATION SCHEME-- K-Fold Cross Validation (k={self.k_folds}) is selected.")
        kfold = KFold(n_splits=self.k_folds, shuffle=True)
        folds_dataloader = {}
        for i, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            train_dataset = torch.utils.data.Subset(self.dataset, train_idx)
            if self.apply_augmentation:
                train_dataset = apply_augmentation_to_spectrograms(train_dataset, aug_to_apply=self.augmentation_methods, time_mask_param=self.augmentation_time_max_param, freq_mask_param=self.augmentation_freq_max_param, k_fold_index=i+1)
            val_dataset = torch.utils.data.Subset(self.dataset, val_idx)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            folds_dataloader[i] = tuple((train_dataloader, val_dataloader))
        return folds_dataloader
    
    def get_dataloaders(self):
        return self.dataloaders
    
if __name__ == "__main__":
    amigos_dataset = AMIGOSDataset(data_path=AMIGOS_FILES_DIR, metadata_path=AMIGOS_METADATA_FILE)
    dataloaders = EEG_dataloader(dataset=amigos_dataset, validation_scheme="LOOCV").get_dataloaders()

    # Just for debugging purposes
    subject_id = 2
    dataloader_test = dataloaders[subject_id]
    train_dataloader, val_dataloader = dataloader_test
    is_train_contain_subject = False
    is_val_contain_subject = True
    for batch in train_dataloader:
        for el in batch['subject_id']:
            if el == subject_id:
                is_train_contain_subject = True
    print("Train: " + str(is_train_contain_subject))
    for batch in val_dataloader:
        for el in batch['subject_id']:
            if el != subject_id:
                print(el)
                is_val_contain_subject = False
    print("Val: " + str(is_val_contain_subject))
    ###
