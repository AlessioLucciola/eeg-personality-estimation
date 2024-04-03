from config import RANDOM_SEED, BATCH_SIZE, VALIDATION_SCHEME, KFOLDCV, SPLIT_RATIO
from datasets.EEG_classification_dataset import EEGClassificationDataset
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader
from typing import Optional
import torch

# TO DO: Remove from here
from config import AMIGOS_FILES_DIR, AMIGOS_METADATA_FILE
from datasets.AMIGOS_dataset import AMIGOSDataset

class EEG_dataloader(DataLoader):
    def __init__(self,
                 dataset: EEGClassificationDataset,
                 seed: int = RANDOM_SEED,
                 batch_size: int = BATCH_SIZE,
                 validation_scheme: str = VALIDATION_SCHEME,
                 k_folds: Optional[int] = KFOLDCV,
                 split_ratio: Optional[float] = SPLIT_RATIO
                ):
        self.dataset = dataset
        self.seed = seed
        self.batch_size = batch_size
        self.validation_scheme = validation_scheme
        self.k_folds = k_folds
        self.split_ratio = split_ratio

        if self.validation_scheme == "LOOCV":
            self.dataloaders = self.loocv()
        elif self.validation_scheme == "K-FOLDCV":
            self.dataloaders = self.kfold()
        elif self.validation_scheme == "SPLIT":
            self.dataloaders = self.split()
        else:
            raise ValueError(f"Validation scheme {self.validation_scheme} is not supported.")
        
    def split(self):
        print(f"--VALIDATION SCHEME-- Split is selected with test_size of {self.split_ratio*100}%.")
        train_df, test_df = train_test_split(self.dataset, test_size=self.split_ratio, random_state=self.seed)
        train_dataloader = DataLoader(train_df, batch_size=self.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_df, batch_size=self.batch_size, shuffle=True)
        return tuple((train_dataloader, test_dataloader))
    
    # Leave-One-Out Cross Validation (LOOCV) based on subject_id
    def loocv(self):
        print("--VALIDATION SCHEME-- Leave-One-Out Cross Validation (LOOCV) is selected.")
        loo_dataloaders = {}
        for i, subject_id in enumerate(self.dataset.subject_ids):
            train_idx = [j for j in range(len(self.dataset.windows)) if self.dataset.windows[j]['subject_id'] != subject_id] # All windows except the one with the subject_id
            val_idx = [j for j in range(len(self.dataset.windows)) if self.dataset.windows[j]['subject_id'] == subject_id] # The window with the subject_id
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            train_dataloader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_subsampler, batch_size=self.batch_size, shuffle=True)
            loo_dataloaders[i] = tuple((train_dataloader, val_dataloader))
        return loo_dataloaders
    
    # K-Fold Cross Validation
    def kfold(self):
        print(f"--VALIDATION SCHEME-- K-Fold Cross Validation (k={self.k_folds}) is selected.")
        kfold = KFold(n_splits=self.k_folds, shuffle=True)
        folds_dataloader = {}
        for i, (train_idx, val_idx) in enumerate(kfold.split(self.dataset)):
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            train_dataloader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_subsampler, batch_size=self.batch_size, shuffle=True)
            folds_dataloader[i] = tuple((train_dataloader, val_dataloader))
        return folds_dataloader
    
if __name__ == "__main__":
    amigos_dataset = AMIGOSDataset(data_path=AMIGOS_FILES_DIR, metadata_path=AMIGOS_METADATA_FILE)
    dataloader = EEG_dataloader(dataset=amigos_dataset, validation_scheme="SPLIT")
    print(dataloader.dataloaders)
