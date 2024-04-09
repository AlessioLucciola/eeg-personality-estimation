from config import DATASET_TO_USE, RANDOM_SEED, BATCH_SIZE, VALIDATION_SCHEME, ELECTRODES, SAMPLING_RATE, MELS, MELS_WINDOW_SIZE, MELS_WINDOW_STRIDE, MELS_MIN_FREQ, MELS_MAX_FREQ, DROPOUT_P, LEARNING_RATE, REG
from dataloaders.EEG_classification_dataloader import EEG_dataloader
from utils.utils import instantiate_dataset, set_seed, select_device
from models.resnet18 import ResNet18
import pytorch_lightning as pl

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    dataset = instantiate_dataset(DATASET_TO_USE)
    dataloaders = EEG_dataloader(dataset=dataset, seed=RANDOM_SEED, batch_size=BATCH_SIZE, validation_scheme=VALIDATION_SCHEME).get_dataloaders()
    model = ResNet18(in_channels=len(ELECTRODES),
                     sampling_rate=SAMPLING_RATE,
                     labels=dataset.labels,
                     labels_classes=dataset.labels_classes,
                     mels=MELS,
                     mel_window_size=MELS_WINDOW_SIZE,
                     mel_window_stride=MELS_WINDOW_STRIDE,
                     mel_min_freq=MELS_MIN_FREQ,
                     mel_max_freq=MELS_MAX_FREQ,
                     dropout_p=DROPOUT_P,
                     learning_rate=LEARNING_RATE,
                     weight_decay=REG,
                     device=device
                     )
    
if __name__ == "__main__":
    main()