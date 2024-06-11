from shared.constants import validation_schemes, supported_datasets, optimizers, positional_encodings, merge_mels_typologies, discretization_methods, augmentation_methods, schedulers, criterions
from typing import List, Union
import os

# General dataset configurations
DATA_DIR: str = "data"
RESULTS_DIR: str = "results"
PLOTS_DIR: str = "plots"
DATASET_TO_USE: str = "AMIGOS" # "AMIGOS" | "ASCERTAIN"
PRINT_DATASET_DEBUG: bool = False # Print debug information during dataset upload if True
MAKE_PLOTS: bool = False # Make plots if True (it takes some time to generate the plots!)

# AMIGOS dataset configurations
AMIGOS_DATASET_DIR: str = os.path.join(DATA_DIR, "amigos")
AMIGOS_FILES_DIR: str = os.path.join(AMIGOS_DATASET_DIR, "files")
AMIGOS_METADATA_FILE: str = os.path.join(AMIGOS_DATASET_DIR, "Participants_Personality.xlsx")
AMIGOS_NUM_CLASSES: int = 2
ELECTRODES: List[str] = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2",
              "P8", "T8", "FC6", "F4", "F8", "AF4"] if DATASET_TO_USE == "AMIGOS" \
            else ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"] # Labels of the electrodes to consider

# ASCERTAIN dataset configurations
ASCERTAIN_DATASET_DIR: str = os.path.join(DATA_DIR, "ascertain")
ASCERTAIN_FILES_DIR: str = os.path.join(ASCERTAIN_DATASET_DIR, "files")
ASCERTAIN_METADATA_FILE: str = os.path.join(ASCERTAIN_DATASET_DIR, "Personality_Details.xls")
ASCERTAIN_NUM_CLASSES: int = 2

# Miscellanous configurations
RANDOM_SEED: int = 42 # Random seed
USE_DML: bool = False # Use DirectML library if True (for AMD GPUs)
USE_WANDB: bool = False # Use Weights & Biases for logging if True
SAVE_RESULTS: bool = True # Save results in JSON files if True
SAVE_MODELS: bool = True # Save models if True

# EEGNet configurations
WINDOWS_SIZE: float = 3 # Size of the sliding window
WINDOWS_STRIDE: float = 3 # Stride of the sliding window
SAMPLING_RATE: int = 256 # Sampling rate of the EEG data
DISCRETIZE_LABELS: bool = True # Discretize the labels if True
DISCRETIZATION_METHOD: str = "personality_mean" # "personality_mean" | "fixed_mean" (only if DISCRETIZE_LABELS is True)
NORMALIZE_DATA: bool = False # Normalize the EEG data if True
DROP_LAST: bool = False # Drop the last window if True (if False, zero-pad the last window)
APPLY_AUGMENTATION: bool = False # Apply data augmentations to mel spectrograms if True
AUGMENTATION_METHODS: List[str] = ["additive_noise", "flipping"] # "spec_augment" | "additive_noise" | "flipping" (only if APPLY_AUGMENTATION is True)
AUGMENTATION_FREQ_MAX_PARAM = 0.35 # Maximum possible length of the frequency mask (only if "spec_augment" is in AUGMENTATION_METHODS and APPLY_AUGMENTATION is True)
AUGMENTATION_TIME_MAX_PARAM = 0.35 # Maximum possible length of the time mask (only if "spec_augment" is in AUGMENTATION_METHODS and APPLY_AUGMENTATION is True)

# Mel spectrogram configurations
MELS: int = 32 # Number of mel bands
MELS_WINDOW_SIZE: float = 0.05 # Size of the window for the mel spectrogram
MELS_WINDOW_STRIDE: float = 0.025 # Stride of the window for the mel spectrogram
MELS_MIN_FREQ: int = 0 # Minimum frequency for the mel spectrogram
MELS_MAX_FREQ: int = 50 # Maximum frequency for the mel spectrogram

# Training configurations
EVALUATE_EACH_LABEL: bool = True # Evaluate each label separately if True
BATCH_SIZE: int = 128 # Batch size
LEARNING_RATE: float = 5e-5 # Learning rate
REG: float = 0.1 # Regularization parameter
EPOCHS: int = 30 # Number of epochs
DROPOUT_P: float = 0.1 # Dropout probability
THRESHOLD: float = 0.5 # Threshold for the binary classification
VALIDATION_SCHEME: str = "LOOCV" # "LOOCV" | "K-FOLDCV" | "SPLIT"
KFOLDCV: int = 3 # Number of folds for K-Fold Cross Validation (only if VALIDATION_SCHEME is "K-FOLDCV")
SPLIT_RATIO: float = 0.2 # Ratio for the train-validation split (only if VALIDATION_SCHEME is "SPLIT")
SUBJECTS_LIMIT: int = 10 # Limit the number of subjects to consider (None for no limit) (only if VALIDATION_SCHEME is "LOOCV")
OPTIMIZER: str = "AdamW" # "Adam" | "AdamW" | "SGD"
SCHEDULER: str = "StepLR" # "StepLR" | "ReduceLROnPlateau" | "CosineAnnealingLR"
CRITERION: str = "TripletMarginLoss" # "BCEWithLogitsLoss" | "CrossEntropyLoss" | "TripletMarginLoss"
SCHEDULER_STEP_SIZE: int = 10 # Step size for the scheduler
SCHEDULER_GAMMA: float = 0.1 # Gamma for the scheduler
LABEL_SMOOTHING_EPSILON: float = 0.0 # Label smoothing (0.0 for no smoothing)

# Resume training configurations
RESUME_TRAINING: bool = False # Resume training if True (specify the path of model to resume and the epoch to start from)
PATH_MODEL_TO_RESUME: str = "ViT_2024-06-06_10-04-34" # Name of the model to resume
RESUME_EPOCH: int = 10 # Epoch to resume
RESUME_FOLD: int = 21 # Fold to resume (only for K-Fold Cross Validation and Leave-One-Out Cross Validation)

# Transformer configurations
NUM_HEADS: int = 4 # Number of heads in the transformer
NUM_ENCODERS: int = 2 # Number of encoder layers in the transformer
NUM_DECODERS: int = 2 # Number of decoder layers in the transformer (only if USE_ENCODER_ONLY is False)
USE_ENCODER_ONLY: bool = True # Use only the encoder part of the transformer if True (no decoder part)
HIDDEN_SIZE: int = 256 # Hidden size in the transformer
POSITIONAL_ENCODING: Union[str, None] = "learnable" # "sinusoidal" | "learnable" | None
USE_LEARNABLE_TOKEN: bool = True # Use learnable token if True (append a learnable token to the input)
MERGE_MELS_TYPOLOGY: str = "channels" # "channels" | "samples" (merge the mel bends leaving the channel dimension or the samples dimension inalterated)

# CNN configurations
USE_PRETRAINED_MODELS: bool = False # Use a pretrained model if True
ADD_DROPOUT_TO_MODEL: bool = False # Add dropout layers to the model if True

# Miscellaneous assertions
assert VALIDATION_SCHEME in validation_schemes, f"{VALIDATION_SCHEME} is not a supported validation scheme."
assert DATASET_TO_USE in supported_datasets, f"{DATASET_TO_USE} is not a supported dataset."
assert OPTIMIZER in optimizers, f"{OPTIMIZER} is not a supported optimizer."
assert SCHEDULER in schedulers, f"{SCHEDULER} is not a supported scheduler."
assert CRITERION in criterions, f"{CRITERION} is not a supported criterion."
assert POSITIONAL_ENCODING in positional_encodings or POSITIONAL_ENCODING is None, f"{POSITIONAL_ENCODING} is not a supported positional encoding."
assert MERGE_MELS_TYPOLOGY in merge_mels_typologies, f"{MERGE_MELS_TYPOLOGY} is not a supported typology for merging the mel bands."
assert DISCRETIZATION_METHOD in discretization_methods, f"{DISCRETIZATION_METHOD} is not a supported discretization method."
assert all([method in augmentation_methods for method in AUGMENTATION_METHODS]), f"{AUGMENTATION_METHODS} is not a supported augmentation method."
assert not APPLY_AUGMENTATION or (APPLY_AUGMENTATION and len(AUGMENTATION_METHODS) > 0), f"Apply augmentation must be True if there are augmentation methods to apply, but got {APPLY_AUGMENTATION} and {AUGMENTATION_METHODS}."
assert 0 <= SPLIT_RATIO <= 1, f"Split ratio must be between 0 and 1, but got {SPLIT_RATIO}."
assert 0 <= DROPOUT_P <= 1, f"Dropout probability must be between 0 and 1, but got {DROPOUT_P}."
assert 0 <= THRESHOLD <= 1, f"Threshold must be between 0 and 1, but got {THRESHOLD}."
assert 0 <= LABEL_SMOOTHING_EPSILON < 1, f"Label smoothing must be between 0 and 1, but got {LABEL_SMOOTHING_EPSILON}."
assert 0 <= REG, f"Regularization parameter must be positive, but got {REG}."
assert 0 < BATCH_SIZE, f"Batch size must be positive, but got {BATCH_SIZE}."
assert 0 < LEARNING_RATE, f"Learning rate must be positive, but got {LEARNING_RATE}."
assert 0 < EPOCHS, f"Number of epochs must be positive, but got {EPOCHS}."
assert 0 < MELS, f"Number of mel bands must be positive, but got {MELS}."
assert 0 < MELS_WINDOW_SIZE, f"Size of the window for the mel spectrogram must be positive, but got {MELS_WINDOW_SIZE}."
assert 0 < MELS_WINDOW_STRIDE, f"Stride of the window for the mel spectrogram must be positive, but got {MELS_WINDOW_STRIDE}."
assert 0 <= MELS_MIN_FREQ, f"Minimum frequency for the mel spectrogram must be non-negative, but got {MELS_MIN_FREQ}."
assert 0 < MELS_MAX_FREQ, f"Maximum frequency for the mel spectrogram must be positive, but got {MELS_MAX_FREQ}."
assert 0 < WINDOWS_SIZE, f"Size of the sliding window must be positive, but got {WINDOWS_SIZE}."
assert 0 < WINDOWS_STRIDE, f"Stride of the sliding window must be positive, but got {WINDOWS_STRIDE}."
assert 0 < SAMPLING_RATE, f"Sampling rate of the EEG data must be positive, but got {SAMPLING_RATE}."
assert 0 < NUM_HEADS, f"Number of heads in the transformer must be positive, but got {NUM_HEADS}."
assert 0 < NUM_ENCODERS, f"Number of encoder layers in the transformer must be positive, but got {NUM_ENCODERS}."
assert 0 <= NUM_DECODERS, f"Number of decoder layers in the transformer must be non-negative, but got {NUM_DECODERS}."
assert 0 < HIDDEN_SIZE, f"Hidden size in the transformer must be positive, but got {HIDDEN_SIZE}."
assert SUBJECTS_LIMIT is None or SUBJECTS_LIMIT > 0, f"Limit of subjects must be positive or None, but got {SUBJECTS_LIMIT}."
assert ELECTRODES, f"List of electrodes cannot be empty."
assert isinstance(DISCRETIZE_LABELS, bool), f"Discretize labels must be a boolean, but got {DISCRETIZE_LABELS}."
assert isinstance(NORMALIZE_DATA, bool), f"Normalize data must be a boolean, but got {NORMALIZE_DATA}."
assert isinstance(DROP_LAST, bool), f"Drop last must be a boolean, but got {DROP_LAST}."
assert isinstance(PRINT_DATASET_DEBUG, bool), f"Print dataset debug must be a boolean, but got {PRINT_DATASET_DEBUG}."
assert isinstance(MAKE_PLOTS, bool), f"Make plots must be a boolean, but got {MAKE_PLOTS}."
assert isinstance(USE_DML, bool), f"Use DirectML must be a boolean, but got {USE_DML}."
assert isinstance(USE_WANDB, bool), f"Use Weights & Biases must be a boolean, but got {USE_WANDB}."
assert isinstance(SAVE_RESULTS, bool), f"Save results must be a boolean, but got {SAVE_RESULTS}."
assert isinstance(SAVE_MODELS, bool), f"Save models must be a boolean, but got {SAVE_MODELS}."
assert isinstance(RANDOM_SEED, int), f"Random seed must be an integer, but got {RANDOM_SEED}."
assert isinstance(RESUME_TRAINING, bool), f"Resume training must be a boolean, but got {RESUME_TRAINING}."
assert isinstance(RESUME_EPOCH, int), f"Epoch to resume must be an integer, but got {RESUME_EPOCH}."
assert isinstance(PATH_MODEL_TO_RESUME, str) or isinstance(PATH_MODEL_TO_RESUME, None), f"Path of the model to resume must be a string or None, but got {PATH_MODEL_TO_RESUME}."
assert isinstance(VALIDATION_SCHEME, str), f"Validation scheme must be a string, but got {VALIDATION_SCHEME}."
assert isinstance(KFOLDCV, int), f"Number of folds for K-Fold Cross Validation must be an integer, but got {KFOLDCV}."
assert isinstance(SPLIT_RATIO, float), f"Split ratio must be a float, but got {SPLIT_RATIO}."
assert isinstance(OPTIMIZER, str), f"Optimizer must be a string, but got {OPTIMIZER}."
assert isinstance(BATCH_SIZE, int), f"Batch size must be an integer, but got {BATCH_SIZE}."
assert isinstance(LEARNING_RATE, float), f"Learning rate must be a float, but got {LEARNING_RATE}."
assert isinstance(REG, float), f"Regularization parameter must be a float, but got {REG}."
assert isinstance(EPOCHS, int), f"Number of epochs must be an integer, but got {EPOCHS}."
assert isinstance(DROPOUT_P, float), f"Dropout probability must be a float, but got {DROPOUT_P}."
assert isinstance(RESULTS_DIR, str), f"Results directory must be a string, but got {RESULTS_DIR}."
assert isinstance(PLOTS_DIR, str), f"Plots directory must be a string, but got {PLOTS_DIR}."
assert isinstance(DATA_DIR, str), f"Data directory must be a string, but got {DATA_DIR}."
assert isinstance(DATASET_TO_USE, str), f"Dataset to use must be a string, but got {DATASET_TO_USE}."
assert isinstance(AMIGOS_DATASET_DIR, str), f"AMIGOS dataset directory must be a string, but got {AMIGOS_DATASET_DIR}."
assert isinstance(AMIGOS_FILES_DIR, str), f"AMIGOS files directory must be a string, but got {AMIGOS_FILES_DIR}."
assert isinstance(AMIGOS_METADATA_FILE, str), f"AMIGOS metadata file must be a string, but got {AMIGOS_METADATA_FILE}."
assert isinstance(AMIGOS_NUM_CLASSES, int), f"AMIGOS number of classes must be an integer, but got {AMIGOS_NUM_CLASSES}."

if RESUME_TRAINING:
    assert EPOCHS >= RESUME_EPOCH, f"The epoch to resume must be less or equal than the total number of epochs to reach, but got {RESUME_EPOCH} and {EPOCHS}."


