from config import SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, EPOCHS
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from utils.utils import save_results, save_model, save_configurations
from sklearn.metrics import roc_auc_score
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import wandb
import copy

def train_eval_loop(device,
                    dataloaders,
                    model,
                    config,
                    optimizer,
                    scheduler,
                    criterion,
                    resume=False):

    if resume:
        data_name = PATH_MODEL_TO_RESUME
        if config["use_wandb"]:
            runs = wandb.api.runs("personality_estimation", filters={"name": data_name})
            if runs:
                run_id = runs[0]["id"]
                wandb.init(
                    project="personality_estimation",
                    id=run_id,
                    resume="allow",
                )
            else:
                print("--WANDB-- Temptative to resume a non-existing run. Starting a new one.")
                wandb.init(
                    project="personality_estimation",
                    config=config,
                    resume=resume,
                    name=data_name
                )
    else:
        # Definition of the parameters to create folders where to save data (plots and models)
        current_datetime = datetime.now()
        current_datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
        data_name = f"{config['architecture']}_{current_datetime_str}"

        if SAVE_RESULTS:
            # Save configurations in JSON
            save_configurations(data_name, config)

        if config["use_wandb"]:
            wandb.init(
                project="personality_estimation",
                config=config,
                resume=resume,
                name=data_name
            )
    if config["validation_scheme"] == "SPLIT":
        train_loader, val_loader = dataloaders

    training_total_step = len(train_loader)
    best_model = None
    best_accuracy = None
    for epoch in range(RESUME_EPOCH if resume else 0, EPOCHS):
        model.train()
        epoch_tr_preds = torch.tensor([])
        epoch_tr_labels = torch.tensor([])
        epoch_tr_outputs = torch.tensor([])
        epoch_tr_loss = 0
        for _, tr_batch in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            tr_data, tr_labels = tr_batch['eeg_data'], tr_batch['labels']
            tr_data = tr_data.to(device)
            tr_labels = tr_labels.to(device)
            tr_outputs = model(tr_data) # Prediction
            #print(tr_outputs, tr_labels)
            tr_loss = criterion(tr_outputs, tr_labels)
            epoch_tr_loss = epoch_tr_loss + tr_loss.item()
            epoch_tr_outputs = torch.cat((epoch_tr_outputs, tr_outputs), 0)

            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()

            with torch.no_grad():
                threshold = 0.5
                tr_preds = (tr_outputs > threshold)
                epoch_tr_preds = torch.cat((epoch_tr_preds, tr_preds), 0)
                epoch_tr_labels = torch.cat((epoch_tr_labels, tr_labels), 0)
        
        with torch.no_grad():
            epoch_tr_preds = epoch_tr_preds.long().cpu().numpy()
            epoch_tr_labels = epoch_tr_labels.long().cpu().numpy()
            tr_accuracy = accuracy_score(epoch_tr_labels, epoch_tr_preds) * 100
            tr_recall = recall_score(epoch_tr_labels, epoch_tr_preds, average='macro') * 100
            tr_precision = precision_score(epoch_tr_labels, epoch_tr_preds, average='macro') * 100
            tr_f1 = f1_score(epoch_tr_labels, epoch_tr_preds, average='macro') * 100
            tr_auroc = []
            for label_idx in range(epoch_tr_labels.shape[1]):
                label_auroc = roc_auc_score(epoch_tr_labels[:, label_idx], epoch_tr_preds[:, label_idx])
                tr_auroc.append(label_auroc)
            tr_auroc = np.mean(tr_auroc) * 100

            print('Training -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
                .format(epoch+1, EPOCHS, epoch_tr_loss/training_total_step, tr_accuracy, tr_recall, tr_precision, tr_f1, tr_auroc))

        if config["use_wandb"]:
            wandb.log({"Training Loss": epoch_tr_loss/training_total_step})
            wandb.log({"Training Accuracy": tr_accuracy})
            wandb.log({"Training Recall": tr_recall})
            wandb.log({"Training Precision": tr_precision})
            wandb.log({"Training F1": tr_f1})
            wandb.log({"Training AUROC": tr_auroc})

        model.eval()
        with torch.no_grad():
            val_total_step = len(val_loader)
            epoch_val_preds = torch.tensor([]).to(device)
            epoch_val_labels = torch.tensor([]).to(device)
            epoch_val_outputs = torch.tensor([]).to(device)
            epoch_val_loss = 0
            for _, val_batch in enumerate(tqdm(val_loader, desc="Validation", leave=False)):
                val_data, val_labels = val_batch['eeg_data'], val_batch['labels']
                val_data = val_data.to(device)
                val_labels = val_labels.to(device)
                val_outputs = model(val_data)
                epoch_val_outputs = torch.cat((epoch_val_outputs, val_outputs), 0)

                val_loss = criterion(val_outputs, val_labels)
                epoch_val_loss = epoch_val_loss + val_loss.item()

                threshold = 0.5
                val_preds = (val_outputs > threshold).long()
                epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
                epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)

            epoch_val_preds = epoch_val_preds.cpu().numpy()
            epoch_val_labels = epoch_val_labels.cpu().numpy()
            val_accuracy = accuracy_score(epoch_val_labels, epoch_val_preds) * 100
            val_recall = recall_score(epoch_val_labels, epoch_val_preds, average='macro') * 100
            val_precision = precision_score(epoch_val_labels, epoch_val_preds, average='macro') * 100
            val_f1 = f1_score(epoch_val_labels, epoch_val_preds, average='macro') * 100
            val_auroc = []
            for label_idx in range(epoch_val_labels.shape[1]):
                label_auroc = roc_auc_score(epoch_val_labels[:, label_idx], epoch_val_preds[:, label_idx])
                val_auroc.append(label_auroc)
            val_auroc = np.mean(val_auroc) * 100

            if config["use_wandb"]:
                wandb.log({"Validation Loss": epoch_val_loss/val_total_step})
                wandb.log({"Validation Accuracy": val_accuracy})
                wandb.log({"Validation Recall": val_recall})
                wandb.log({"Validation Precision": val_precision})
                wandb.log({"Validation F1": val_f1})
                wandb.log({"Validation AUROC": val_auroc})
            print('Validation -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
                  .format(epoch+1, EPOCHS, epoch_val_loss/val_total_step, val_accuracy, val_recall, val_precision, val_f1, val_auroc))

            if best_accuracy is None or val_accuracy < best_accuracy:
                best_accuracy = val_accuracy
                best_model = copy.deepcopy(model)
            current_results = {
                'epoch': epoch+1,
                'training_loss': epoch_tr_loss/training_total_step,
                'training_accuracy': tr_accuracy,
                'training_recall': tr_recall,
                'training_precision': tr_precision,
                'training_f1': tr_f1,
                'training_auroc': tr_auroc,
                'validation_loss': epoch_val_loss/val_total_step,
                'validation_accuracy': val_accuracy,
                'validation_recall': val_recall,
                'validation_precision': val_precision,
                'validation_f1': val_f1,
                'validation_auroc': val_auroc
            }
            if SAVE_RESULTS:
                save_results(data_name, current_results)
            if SAVE_MODELS:
                save_model(data_name, model, epoch)
            if epoch == EPOCHS-1 and SAVE_MODELS:
                save_model(data_name, best_model, epoch=None, is_best=True)

        #scheduler.step()