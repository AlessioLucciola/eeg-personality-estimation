from datasets.EEG_classification_dataset import EEGClassificationDataset
from torch.utils.data import Dataset
from typing import List
import random

class EEGTripletDataset(Dataset):
    def __init__(self,
            dataset: EEGClassificationDataset
        ):
        self.dataset = dataset
        
        # Precompute the indices of samples by subject
        self.subject_samples = {}
        for idx, sample in enumerate(self.dataset):
            subject_id = sample['subject_id']
            if subject_id not in self.subject_samples:
                self.subject_samples[subject_id] = []
            self.subject_samples[subject_id].append(idx)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,
            index: int
        ):
        # Select an anchor sample
        anchor = self.dataset[index]
        anchor_subject_id = anchor['subject_id']

        # Select a positive sample from the same subject
        positive_index = random.choice(self.subject_samples[anchor_subject_id])
        positive = self.dataset[positive_index]
        
        # Select a negative sample from a different subject
        negative_subject_id = random.choice([k for k in self.subject_samples.keys() if k != anchor_subject_id])
        negative_index = random.choice(self.subject_samples[negative_subject_id])
        negative = self.dataset[negative_index]

        return anchor, positive, negative