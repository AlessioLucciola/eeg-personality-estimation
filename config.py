from shared.constants import validation_schemes
import os

# General dataset configurations
DATA_DIR = "data"
RESULTS_DIR = "results"

# AMIGOS dataset configurations
AMIGOS_DATASET_DIR = os.path.join(DATA_DIR, "amigos")
AMIGOS_FILES_DIR = os.path.join(AMIGOS_DATASET_DIR, "files")
AMIGOS_METADATA_FILE = os.path.join(AMIGOS_DATASET_DIR, "Participants_Personality.xlsx")
AMIGOS_NUM_CLASSES = 2

# Miscellanous configurations
RANDOM_SEED = 42 # Random seed
USE_DML = False # Use DirectML library if True (for AMD GPUs)
USE_WANDB = False # Use Weights & Biases for logging if True

# EEGNet configurations
WINDOWS_SIZE = 3 # Size of the sliding window
WINDOWS_STRIDE = 1 # Stride of the sliding window
SAMPLING_RATE = 128 # Sampling rate of the EEG data
ELECTRODES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
              "P8", "T8", "FC6", "F4", "F8", "AF4"] # Labels of the electrodes to consider
DISCRETIZE_LABELS = True # Discretize the labels if True
NORMALIZE_DATA = True # Normalize the EEG data if True
DROP_LAST = False # Drop the last window if True

# Training configurations
BATCH_SIZE = 32 # Batch size
LEARNING_RATE = 0.001 # Learning rate
EPOCHS = 100 # Number of epochs
VALIDATION_SCHEME = "K-FOLDCV" # "LOOCV" | "K-FOLDCV" | "SPLIT"
KFOLDCV = 5 # Number of folds for K-Fold Cross Validation
SPLIT_RATIO = 0.2 # Ratio for the train-validation split

# Miscellaneous assertions
assert VALIDATION_SCHEME in validation_schemes, f"{VALIDATION_SCHEME} is not a supported validation scheme."