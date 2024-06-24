from config import AMIGOS_NUM_CLASSES, AMIGOS_FILES_DIR, AMIGOS_METADATA_FILE, PRINT_DATASET_DEBUG
from datasets.EEG_classification_dataset import EEGClassificationDataset
from shared.constants import amigos_labels
from collections import deque
from tqdm import tqdm
from scipy import io
import numpy as np
import pandas as pd
import os

class AMIGOSDataset(EEGClassificationDataset):
    def __init__(self,
                 data_path: str,
                 metadata_path: str,
                 apply_label_discretization: bool,
                 discretization_method: str
                ):
        super(AMIGOSDataset, self).__init__(
            data_path=data_path,
            metadata_path=metadata_path,
            apply_label_discretization=apply_label_discretization,
            discretization_method=discretization_method,
            dataset_name="AMIGOS",
            subject_ids=None,
            labels=amigos_labels,
            labels_classes=AMIGOS_NUM_CLASSES
        )
    
    def load_data(self):
        metadata_df = self.upload_metadata() # Upload the metadata file
        if self.apply_label_discretization:
            metadata_df = self.discretize_labels(metadata_df, discretization_method=self.discretization_method) # Discretize the personality traits
        eeg_df = self.upload_eeg_data() # Upload the EEG data
        metadata_df, eeg_df = self.check_metadata_validity(metadata_df, eeg_df) # Check if the metadata file is valid
        
        eegs_list = deque()
        labels_list = deque()
        subjects_list = deque()
 
        for egg_file, eeg_data in tqdm(eeg_df.items(), desc="Parsing EEG data..", unit="subject", leave=False):
            _, subject_id = self.parse_amigos_file_names(egg_file) # Parse the file name to get the subject ID
            subjects_list.append(subject_id) # Append the subject ID
            eegs_list.append(eeg_data) # Append the EEG data of the subject
            labels_dict = metadata_df[metadata_df['UserID'] == subject_id].iloc[0, 1:].to_dict() # Extract the personality traits of the subject
            mapped_labels_dict = {amigos_labels[key]: value for key, value in labels_dict.items()} # Map column names to their corresponding integer values
            labels_list.append(mapped_labels_dict) # Append the personality traits of the subject
        del metadata_df, eeg_df # Remove the metadata and EEG data to free up memory
        return list(eegs_list), list(labels_list), list(subjects_list)

    
    def upload_metadata(self):
        # Upload the metadata file with the subject IDs and associated personality traits
        metadata_df = pd.read_excel(io=self.metadata_path, sheet_name="Personalities", nrows=6, header=None).T
        metadata_df.columns = metadata_df.iloc[0] # Set the first row as the column names
        metadata_df = metadata_df[1:] # Remove the first row
        metadata_df['UserID'] = metadata_df['UserID'].astype(int)  # Convert 'UserID' column to integer
        subjects = list(metadata_df['UserID']) # First row contains the subject IDs
        self.subject_ids = subjects # Set the subject IDs
        del subjects # Remove the subjects list to free up memory
        return metadata_df
    
    def discretize_labels(self, metadata_df, discretization_method):
        print("--DATASET-- Discretize labels parameters set to True. Discretizing personality traits..")
        traits = metadata_df.columns[1:] # Extract the personality traits
        if discretization_method == "fixed_mean":
            print("--DATASET-- Discretizing personality traits based on fixed mean value of the dataset (that is 4)")
            # Discretize the personality traits based on fixed mean value of the dataset (that is 4)
            for trait in traits:
                metadata_df[trait] = metadata_df[trait].apply(lambda x: 1 if x > 4 else 0) # Discretize the personality trait based on the mean value
            del traits # Remove the traits list to free up memory
            return metadata_df
        elif discretization_method == "personality_mean":
            print("--DATASET-- Discretizing personality traits based on their mean value")
            # Discretize the personality traits based on their mean value
            for trait in traits:
                mean = metadata_df[trait].mean() # Calculate the mean value of the personality trait
                assert mean >= 1 and mean <= 7 # Check if the mean is within the range of the personality trait (it must be a value between 1 and 7)
                metadata_df[trait] = metadata_df[trait].apply(lambda x: 1 if x > mean else 0) # Discretize the personality trait based on the mean value
            del traits # Remove the traits list to free up memory
            return metadata_df
        else:
            raise ValueError(f"Unknown discretization method: {discretization_method}")

    def upload_eeg_data(self):
        # Upload the EEG data
        electrodes_data = {}
        for file in tqdm(os.listdir(self.data_path), desc="Uploading EEG data..", unit="file", leave=False):
            if file.endswith(".mat"):
                eeg_data = io.loadmat(os.path.join(self.data_path, file), simplify_cells=True)
                trials = [e[:, :14].astype(np.float32) for e in eeg_data["joined_data"] if e.size != 0]
                electrodes_data[file] = trials
        return electrodes_data
    
    def check_metadata_validity(self, metadata_df, eeg_df):
        # Check if the metadata file is valid (all subject IDs are unique, all subject have personality traits)
        missing_subjects = []
        files_to_remove = []
        for egg_file in eeg_df.keys():
            _, subject_id = self.parse_amigos_file_names(egg_file)
            if subject_id not in metadata_df['UserID'].values: # Check if the subject ID is in the metadata file, if not then remove the associated file
                missing_subjects.append(subject_id) # Add the missing subject ID to the list
                metadata_df = metadata_df[metadata_df['UserID'] != subject_id] # Remove the subject ID from the metadata file
                files_to_remove.append(egg_file) # Add the file to the list of files to remove
        if len(missing_subjects) > 0:
            if PRINT_DATASET_DEBUG:
                print(f"--DATASET-- Missing personality traits of these subjects (the associated files won't be considered): {missing_subjects}")
            for file in files_to_remove: # Remove the files that are not in the metadata file
                del eeg_df[file]
        else:
            if PRINT_DATASET_DEBUG:
                print("--DATASET-- All subjects have personality traits")
        del missing_subjects, files_to_remove # Remove the lists to free up memory
        return metadata_df, eeg_df

    def parse_amigos_file_names(self, file):
        # Parse the file names of the AMIGOS dataset
        file_info = file.split("_")
        dataset_type = file_info[1] # It can be original or preprocessed
        subject_id = int(file_info[2].split("P")[1].split(".")[0]) # Extract the subject ID from the file name
        return dataset_type, subject_id

if __name__ == "__main__":
    dataset = AMIGOSDataset(data_path=AMIGOS_FILES_DIR, metadata_path=AMIGOS_METADATA_FILE)