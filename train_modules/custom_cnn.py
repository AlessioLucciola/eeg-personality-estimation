from train_modules.train_loops.train_loop_kfold_loo import train_eval_loop as train_eval_loop_kfold_loo
from train_modules.train_loops.train_loop_split import train_eval_loop as train_eval_loop_split
from utils.train_utils import count_model_MACs, get_criterion, get_model_params, get_optimizer, get_scheduler
from utils.utils import get_configurations, instantiate_dataset, set_seed, select_device
from dataloaders.EEG_classification_dataloader import EEG_dataloader
from models.custom_cnn import CustomCNN
from config import *
import torch

def main():
    resumed_configuration = None
    if RESUME_TRAINING:
        resumed_configuration = get_configurations(PATH_MODEL_TO_RESUME)

    seed = resumed_configuration["seed"] if resumed_configuration != None else RANDOM_SEED
    set_seed(seed)
    device = select_device()
    dataset = instantiate_dataset(
        dataset_name=resumed_configuration["dataset"] if resumed_configuration != None else DATASET_TO_USE,
        apply_label_discretization=resumed_configuration["discretize_labels"] if resumed_configuration != None else DISCRETIZE_LABELS,
        discretization_method=resumed_configuration["discretization_method"] if resumed_configuration != None else DISCRETIZATION_METHOD
    )
    dataloader = EEG_dataloader(
        dataset=dataset,
        seed=seed,
        batch_size=resumed_configuration["batch_size"] if resumed_configuration != None else BATCH_SIZE,
        validation_scheme=resumed_configuration["validation_scheme"] if resumed_configuration != None else VALIDATION_SCHEME,
        apply_augmentation=resumed_configuration["is_data_augmented"] if resumed_configuration != None else APPLY_AUGMENTATION,
        augmentation_methods=resumed_configuration["augmentation_methods"] if resumed_configuration != None else AUGMENTATION_METHODS,
        augmentation_freq_max_param=resumed_configuration["augmentation_freq_max_param"] if resumed_configuration != None else AUGMENTATION_FREQ_MAX_PARAM,
        augmentation_time_max_param=resumed_configuration["augmentation_time_max_param"] if resumed_configuration != None else AUGMENTATION_TIME_MAX_PARAM
    )
    dataloaders = dataloader.get_dataloaders()
    
    model = CustomCNN(
        in_channels=len(resumed_configuration["electrodes"]) if resumed_configuration != None else len(ELECTRODES),
        labels=dataset.labels,
        labels_classes=dataset.labels_classes,
        dropout_p=resumed_configuration["dropout_p"] if resumed_configuration != None else DROPOUT_P,
        device=device
    ).to(device)

    if RESUME_TRAINING:
        model_path = f"{RESULTS_DIR}/{PATH_MODEL_TO_RESUME}/models/personality_estimation_{RESUME_EPOCH}.pt" if resumed_configuration["validation_scheme"] == "SPLIT" else f"{RESULTS_DIR}/{PATH_MODEL_TO_RESUME}/models/personality_estimation_fold_{RESUME_FOLD}_epoch_{RESUME_EPOCH}.pt"
        model.load_state_dict(torch.load(model_path))

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
            "architecture": "CustomCNN",
            "model_params": get_model_params(model),
            "labels": dataset.labels,
            "discretize_labels": DISCRETIZE_LABELS,
            "discretization_method": DISCRETIZATION_METHOD,
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
            "dropout_p": DROPOUT_P,
            "is_data_augmented": APPLY_AUGMENTATION,
            "augmentation_methods": AUGMENTATION_METHODS,
            "augmentation_freq_max_param": AUGMENTATION_FREQ_MAX_PARAM,
            "augmentation_time_max_param": AUGMENTATION_TIME_MAX_PARAM,
            "use_wandb": USE_WANDB,
            "use_dml": USE_DML
        }
    else:
        config = resumed_configuration
    
    if config["validation_scheme"] == "SPLIT":
        if resumed_configuration is None:
            input_dummy_data = next(iter(dataloaders[0]))['spectrogram'][0].unsqueeze(0).to(device) #Take one element from the first fold (for MACs calculation)
            macs = count_model_MACs(model, input_dummy_data)
            config["macs"] = macs
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
        if resumed_configuration is None:
            fold_dataloaders = list(dataloaders.items())
            input_dummy_data = next(iter(fold_dataloaders[0][1][0]))['spectrogram'][0].unsqueeze(0).to(device) #Take one element from the first fold (for MACs calculation)
            macs = count_model_MACs(model, input_dummy_data)
            config["macs"] = macs
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