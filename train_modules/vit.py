from utils.utils import get_configurations, instantiate_dataset, set_seed, select_device
from utils.train_utils import get_criterion, get_optimizer, get_scheduler
from dataloaders.EEG_classification_dataloader import EEG_dataloader
from train_modules.train_loops.train_loop_split import train_eval_loop as train_eval_loop_split
from train_modules.train_loops.train_loop_kfold_loo import train_eval_loop as train_eval_loop_kfold_loo
from models.vit import ViT
from config import *
import torch

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    dataset = instantiate_dataset(DATASET_TO_USE)
    dataloader = EEG_dataloader(dataset=dataset, seed=RANDOM_SEED, batch_size=BATCH_SIZE, validation_scheme=VALIDATION_SCHEME)
    dataloaders = dataloader.get_dataloaders()
    model = ViT(in_channels=len(ELECTRODES),
            labels=dataset.labels,
            labels_classes=dataset.labels_classes,
            hidden_size=HIDDEN_SIZE,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            mels=MELS,
            dropout_p=DROPOUT_P,
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
    criterion = get_criterion(
        criterion_name=resumed_configuration["criterion"] if resumed_configuration != None else CRITERION,
        smoothing_factor=resumed_configuration["label_smoothing_epsilon"] if resumed_configuration != None else LABEL_SMOOTHING_EPSILON
    )

    if resumed_configuration == None:
        config = {
            "architecture": "ViT",
            "labels": dataset.labels,
            "discretize_labels": DISCRETIZE_LABELS,
            "num_classes": dataset.labels_classes,
            "evaluate_each_label": EVALUATE_EACH_LABEL,
            "normalize_data": NORMALIZE_DATA,
            "optimizer": OPTIMIZER,
            "criterion": CRITERION,
            "label_smoothing_epsilon": LABEL_SMOOTHING_EPSILON,
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
            "eeg_window_size": WINDOWS_SIZE,
            "eeg_window_stride": WINDOWS_STRIDE,
            "mels": MELS,
            "mel_window_size": MELS_WINDOW_SIZE,
            "mel_window_stride": MELS_WINDOW_STRIDE,
            "mel_min_freq": MELS_MIN_FREQ,
            "mel_max_freq": MELS_MAX_FREQ,
            "transformer_hidden_size": HIDDEN_SIZE,
            "transformer_num_heads": NUM_HEADS,
            "transformer_num_layers": NUM_LAYERS,
            "dropout_p": DROPOUT_P,
            "is_data_augmented": APPLY_AUGMENTATION,
            "use_dml": USE_DML,
            "use_wandb": USE_WANDB 
        }
    
    if config["validation_scheme"] == "SPLIT":
        if resumed_configuration != None:
            config["split_ratio"] = dataloader.split_ratio

        train_eval_loop_split(device=device,
                            dataloaders=dataloaders,
                            model=model,
                            config=resumed_configuration if resumed_configuration != None else config,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            criterion=criterion,
                            resume=RESUME_TRAINING
                        )
    else:
        if resumed_configuration != None:
            config["k_folds"] = dataloader.k_folds if config["validation_scheme"] == "K-FOLDCV" else len(dataloader.dataset.subjects_ids)
        
        train_eval_loop_kfold_loo(device=device,
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