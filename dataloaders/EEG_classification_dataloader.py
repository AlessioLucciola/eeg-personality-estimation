from config import RANDOM_SEED, BATCH_SIZE, VALIDATION_SCHEME
from datasets.EEG_classification_dataset import EEGClassificationDataset
from torch.utils.data import DataLoader

class EEG_dataloader(DataLoader):
    def __init__(self,
                 dataset: EEGClassificationDataset,
                 seed: int = RANDOM_SEED,
                 batch_size: int = BATCH_SIZE,
                 validation_scheme: str = VALIDATION_SCHEME
                ):
        super(EEG_dataloader, self).__init__()
        self.seed = seed
        self.batch_size = batch_size
        self.validation_scheme = validation_scheme

        if self.validation_scheme == "LOOCV":
            self.train_dataloader, self.val_dataloader = self.loocv()
        elif self.validation_scheme == "K-FOLDCV":
            self.train_dataloader, self.val_dataloader = self.kfold()
    
    def loocv(self):
        pass

    def kfold(self):
        pass
