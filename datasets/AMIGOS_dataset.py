from config import AMIGOS_NUM_CLASSES, AMIGOS_FILES_DIR, AMIGOS_METADATA_FILE, DISCRETIZE_LABELS
from shared.constants import amigos_labels
from datasets.EEG_classification_dataset import EEGClassificationDataset
from tqdm import tqdm
from scipy import io
import pandas as pd
import os

class AMIGOSDataset(EEGClassificationDataset):
    def __init__(self,
                 data_path: str,
                 metadata_path: str
                ):
        super(AMIGOSDataset, self).__init__(
            data_path=data_path,
            metadata_path=metadata_path,
            dataset_name="AMIGOS",
            subject_ids=None,
            labels=amigos_labels,
            labels_classes=AMIGOS_NUM_CLASSES,
        )
    
    def load_data(self):
        metadata_df = self.upload_metadata()
        # TO DO: Create a function to discretize the labels
        if DISCRETIZE_LABELS:
            metadata_df = self.discretize_labels(metadata_df)

        eeg_df = self.upload_eeg_data() # Upload the EEG data

        # TO DO: Create a function to see if there is consistency between the metadata and the data files
        self.check_metadata_validity() # Check if the metadata file is valid

        return None, None, None
    
    def upload_metadata(self):
        # Upload the metadata file with the subject IDs and associated personality traits
        metadata_df = pd.read_excel(io=self.metadata_path, sheet_name="Personalities", nrows=6, header=None).T
        metadata_df.columns = metadata_df.iloc[0] # Set the first row as the column names
        metadata_df = metadata_df[1:] # Remove the first row
        subjects = list(metadata_df['UserID'].astype(int)) # First row contains the subject IDs
        self.subject_ids = subjects # Set the subject IDs
        return metadata_df
    
    def discretize_labels(self, metadata_df):
        pass

    def upload_eeg_data(self):
        # Upload the EEG data
        electrodes_data = {}
        for file in tqdm(os.listdir(self.data_path), desc="Uploading EEG data..", unit="file", leave=False):
            if file.endswith(".mat"):
                eeg_data = io.loadmat(os.path.join(self.data_path, file))
                electrodes_data[file] = eeg_data["joined_data"][:, :14]
        return electrodes_data
    
    def check_metadata_validity(self):
        # Check if the metadata file is valid (all subject IDs are unique, all subject have personality traits)
        pass

if __name__ == "__main__":
    dataset = AMIGOSDataset(data_path=AMIGOS_FILES_DIR, metadata_path=AMIGOS_METADATA_FILE)
    dataset.load_data()