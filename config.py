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
RANDOM_SEED = 42
USE_DML = False
USE_WANDB = False

# EEGNet configurations
WINDOWS_SIZE = 3
WINDOWS_STRIDE = 1
SAMPLING_RATE = 128
ELECTRODES = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
              "P8", "T8", "FC6", "F4", "F8", "AF4"]
DISCRETIZE_LABELS = True
NORMALIZE_DATA = True
DROP_LAST = False # Drop the last window if True

# Training configurations
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 100
VALIDATION_SCHEME = "LOOCV"
KFOLDCV = 5

# Miscellaneous assertions
assert (VALIDATION_SCHEME in validation_schemes, f"{VALIDATION_SCHEME} is not a supported validation scheme.")