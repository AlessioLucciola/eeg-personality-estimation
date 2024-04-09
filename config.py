from shared.constants import validation_schemes, supported_datasets, optimizers
import os

# General dataset configurations
DATA_DIR = "data"
RESULTS_DIR = "results"
DATASET_TO_USE = "AMIGOS" # "AMIGOS"
PRINT_DATASET_DEBUG = True # Print debug information during dataset upload if True

# AMIGOS dataset configurations
AMIGOS_DATASET_DIR = os.path.join(DATA_DIR, "amigos")
AMIGOS_FILES_DIR = os.path.join(AMIGOS_DATASET_DIR, "files")
AMIGOS_METADATA_FILE = os.path.join(AMIGOS_DATASET_DIR, "Participants_Personality.xlsx")
AMIGOS_NUM_CLASSES = 2

# Miscellanous configurations
RANDOM_SEED = 42 # Random seed
USE_DML = True # Use DirectML library if True (for AMD GPUs)
USE_WANDB = False # Use Weights & Biases for logging if True

# EEGNet configurations
WINDOWS_SIZE = 3 # Size of the sliding window
WINDOWS_STRIDE = 3 # Stride of the sliding window
SAMPLING_RATE = 128 # Sampling rate of the EEG data
ELECTRODES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
              "P8", "T8", "FC6", "F4", "F8", "AF4"] # Labels of the electrodes to consider
DISCRETIZE_LABELS = True # Discretize the labels if True
NORMALIZE_DATA = True # Normalize the EEG data if True
DROP_LAST = False # Drop the last window if True
MELS = 32 # Number of mel bands
MELS_WINDOW_SIZE = 1 # Size of the window for the mel spectrogram
MELS_WINDOW_STRIDE = 0.05 # Stride of the window for the mel spectrogram
MELS_MIN_FREQ = 0 # Minimum frequency for the mel spectrogram
MELS_MAX_FREQ = 50 # Maximum frequency for the mel spectrogram

# Training configurations
BATCH_SIZE = 32 # Batch size
LEARNING_RATE = 0.001 # Learning rate
REG = 0.1 # Regularization parameter
EPOCHS = 100 # Number of epochs
DROPOUT_P = 0.25 # Dropout probability
VALIDATION_SCHEME = "SPLIT" # "LOOCV" | "K-FOLDCV" | "SPLIT"
KFOLDCV = 5 # Number of folds for K-Fold Cross Validation
SPLIT_RATIO = 0.2 # Ratio for the train-validation split
OPTIMIZER = "Adam" # "Adam" | "AdamW" | "SGD"

# Miscellaneous assertions
assert VALIDATION_SCHEME in validation_schemes, f"{VALIDATION_SCHEME} is not a supported validation scheme."
assert DATASET_TO_USE in supported_datasets, f"{DATASET_TO_USE} is not a supported dataset."
assert OPTIMIZER in optimizers, f"{OPTIMIZER} is not a supported optimizer."
