import os

# General dataset configurations
DATA_DIR = "data"
RESULTS_DIR = "results"

# AMIGOS dataset configurations
AMIGOS_DATASET_DIR = os.path.join(DATA_DIR, "AMIGOS")
AMIGOS_METADATA_FILE = os.path.join(AMIGOS_DATASET_DIR, "Participant_Personality.xlsx")
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