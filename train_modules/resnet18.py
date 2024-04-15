from config import DATASET_TO_USE, RANDOM_SEED, BATCH_SIZE, VALIDATION_SCHEME, ELECTRODES, SAMPLING_RATE, MELS, MELS_WINDOW_SIZE, MELS_WINDOW_STRIDE, MELS_MIN_FREQ, MELS_MAX_FREQ, DROPOUT_P, LEARNING_RATE, REG, RESUME_TRAINING, RESULTS_DIR, PATH_MODEL_TO_RESUME, RESUME_EPOCH, LEARNING_RATE, OPTIMIZER, SCHEDULER, SCHEDULER_STEP_SIZE, SCHEDULER_GAMMA, USE_WANDB, THRESHOLD
from utils.utils import get_configurations, instantiate_dataset, set_seed, select_device
from utils.train_utils import get_criterion, get_optimizer, get_scheduler
from dataloaders.EEG_classification_dataloader import EEG_dataloader
from train_modules.train_loops.train_loop import train_eval_loop
from models.resnet18 import ResNet18
import torch

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
                     ).to(device)
    resumed_configuration = None
    if RESUME_TRAINING:
        model.load_state_dict(torch.load(
            f"{RESULTS_DIR}/{PATH_MODEL_TO_RESUME}/models/mi_project_{RESUME_EPOCH}.pt"))
        resumed_configuration = get_configurations(PATH_MODEL_TO_RESUME)
    optimizer = get_optimizer(
        optimizer_name=resumed_configuration["optimizer"] if resumed_configuration != None else OPTIMIZER,
        parameters=model.parameters(),
        lr=resumed_configuration["lr"] if resumed_configuration != None else LEARNING_RATE,
        weight_decay=resumed_configuration["reg"] if resumed_configuration != None else REG
        )
    scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_name=resumed_configuration["scheduler"] if resumed_configuration != None else SCHEDULER,
        step_size=resumed_configuration["scheduler_step_size"] if resumed_configuration != None else SCHEDULER_STEP_SIZE,
        gamma=resumed_configuration["scheduler_gamma"] if resumed_configuration != None else SCHEDULER_GAMMA
        )

    if resumed_configuration == None:
        config = {
            "architecture": "ResNet18",
            "labels": dataset.labels,
            "num_classes": dataset.labels_classes,
            "optimizer": OPTIMIZER,
            "lr": LEARNING_RATE,
            "reg": REG,
            "batch_size": BATCH_SIZE,
            "threshold": THRESHOLD,
            "scheduler": SCHEDULER,
            "scheduler_step_size": SCHEDULER_STEP_SIZE,
            "scheduler_gamma": SCHEDULER_GAMMA,
            "dataset": DATASET_TO_USE,
            "seed": RANDOM_SEED,
            "validation_scheme": VALIDATION_SCHEME,
            "electrodes": ELECTRODES,
            "sampling_rate": SAMPLING_RATE,
            "mels": MELS,
            "mel_window_size": MELS_WINDOW_SIZE,
            "mel_window_stride": MELS_WINDOW_STRIDE,
            "mel_min_freq": MELS_MIN_FREQ,
            "mel_max_freq": MELS_MAX_FREQ,
            "dropout_p": DROPOUT_P,
            "use_wandb": USE_WANDB 
        }

    criterion = get_criterion()
    
    train_eval_loop(device=device,
                    dataloaders=dataloaders,
                    model=model,
                    config=resumed_configuration if resumed_configuration != None else config,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    criterion=criterion,
                    resume=RESUME_TRAINING
                    )
    
if __name__ == "__main__":
    main()