from datasets.EEG_classification_dataset import EEGClassificationDataset
from shared.constants import ascertain_labels
from collections import deque
from tqdm import tqdm
from scipy import io
from config import *
import pandas as pd
import numpy as np
import einops

class ASCERTAINDataset(EEGClassificationDataset):
    def __init__(self,
                 data_path: str,
                 metadata_path: str
                ):
        super(ASCERTAINDataset, self).__init__(
            data_path=data_path,
            metadata_path=metadata_path,
            dataset_name="ASCERTAIN",
            subject_ids=None,
            labels=ascertain_labels,
            labels_classes=ASCERTAIN_NUM_CLASSES,
        )
    
    def load_data(self):
        metadata_df = self.upload_metadata() # Upload the metadata file
        if DISCRETIZE_LABELS:
            metadata_df = self.discretize_labels(metadata_df) # Discretize the personality traits
        eeg_df = self.upload_eeg_data() # Upload the EEG data
        
        eegs_list = deque()
        labels_list = deque()
        subjects_list = deque()

        for eeg_data, subject_id in tqdm(eeg_df, desc="Parsing EEG data..", unit="subject", leave=False):
            subjects_list.append(subject_id) # Append the subject ID
            eegs_list.append(eeg_data) # Append the EEG data of the subject
            labels_dict = metadata_df[metadata_df['Subject ID'] == subject_id].iloc[0, 1:].to_dict() # Extract the personality traits of the subject
            mapped_labels_dict = {ascertain_labels[key]: value for key, value in labels_dict.items()} # Map column names to their corresponding integer values
            labels_list.append(mapped_labels_dict) # Append the personality traits of the subject
        return list(eegs_list), list(labels_list), list(subjects_list)


    def upload_metadata(self):
        # Upload the metadata file with the subject IDs and associated personality traits
        metadata_df = pd.read_excel(io=self.metadata_path, sheet_name="Results", nrows=59, header=None)
        metadata_df.columns = metadata_df.iloc[0] # Set the first row as the column names
        metadata_df = metadata_df[1:] # Remove the first row
        metadata_df['Subject ID'] = metadata_df['Subject ID'].astype(int)  # Convert 'Subject ID' column to integer
        subjects = list(metadata_df['Subject ID']) # First row contains the subject IDs
        self.subject_ids = subjects # Set the subject IDs
        return metadata_df

    def discretize_labels(self, metadata_df):
        # Discretize the personality traits based on their mean value
        traits = metadata_df.columns[1:]
        for trait in traits:
            mean = metadata_df[trait].mean() # Calculate the mean value of the personality trait
            assert mean >= 1 and mean <= 7
            metadata_df[trait] = metadata_df[trait].apply(lambda x: 1 if x > mean else 0) # Discretize the personality trait based on the mean value
        return metadata_df
    
    def upload_eeg_data(self):
        missing_subjects = []
        electrodes_data = []
        for subject_folder in tqdm(os.listdir(self.data_path), desc="Uploading EEG data..", unit="subject", leave=False):
            subject_id = int(subject_folder[-2:]) # Extract the subject ID from the folder name (last two characters)
            if subject_id in self.subject_ids:
                subject_experiments = [] 
                for file in os.listdir(os.path.join(self.data_path, subject_folder)):
                    if file.endswith('.mat'):
                        file_path = os.path.join(self.data_path, subject_folder, file)
                        eeg_data = io.loadmat(file_path, simplify_cells=True)['ThisEEG'].astype(np.float32) # Load the EEG data (only the 'ThisEEG' field is needed
                        eeg_data = einops.rearrange(eeg_data, "c s -> s c")
                        if eeg_data.shape[1] == 9:
                            eeg_data = eeg_data[:, :8]
                        subject_experiments.append(eeg_data)
                electrodes_data.append(tuple((subject_experiments, subject_id)))
            else:
                missing_subjects.append(subject_id)
        if PRINT_DATASET_DEBUG:
            if len(missing_subjects) > 0:
                print(f"--DATASET-- Missing personality traits of these subjects (the associated files won't be considered): {missing_subjects}")
            else:
                print("--DATASET-- All subjects have associated personality traits")

        return electrodes_data

if __name__ == "__main__":
    dataset = ASCERTAINDataset(data_path=ASCERTAIN_FILES_DIR, metadata_path=ASCERTAIN_METADATA_FILE)
    dataset.load_data()
