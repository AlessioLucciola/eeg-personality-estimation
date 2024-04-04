from config import DATASET_TO_USE, RANDOM_SEED, BATCH_SIZE, VALIDATION_SCHEME
from dataloaders.EEG_classification_dataloader import EEG_dataloader
from utils.utils import instantiate_dataset, set_seed, select_device
from torch.profiler import profile, ProfilerActivity
import pytorch_lightning as pl

def main():
    set_seed(RANDOM_SEED)
    device = select_device()
    dataset = instantiate_dataset(DATASET_TO_USE)
    dataloaders = EEG_dataloader(dataset=dataset, seed=RANDOM_SEED, batch_size=BATCH_SIZE, validation_scheme=VALIDATION_SCHEME).get_dataloaders()
    model = None # TO DO: Instantiate model (to be created yet)
    with profile(activities=ProfilerActivity.CPU, record_shapes=True) as prof:
        trainer = pl.Trainer(
            accelerator=device,
            max_epochs=1,
            check_val_every_n_epoch=1,
            logger=False,
            log_every_n_steps=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=False,
            gradient_clip_val=0,
            limit_train_batches=1,
            limit_val_batches=1,
        )
        if VALIDATION_SCHEME == "LOOCV" or VALIDATION_SCHEME == "K-FOLDCV":
            for i, (train_dataloader, val_dataloader) in dataloaders.items():
                print(f"--TRAINING--Training fold {i+1}/{len(dataloaders)}")
                trainer.fit(model, train_dataloaders=val_dataloader, val_dataloaders=val_dataloader)
        elif VALIDATION_SCHEME == "SPLIT":
            train_dataloader, test_dataloader = dataloaders
            trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=test_dataloader)
        else:
            raise ValueError(f"Validation scheme {VALIDATION_SCHEME} is not supported.")
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=8))

if __name__ == "__main__":
    main()