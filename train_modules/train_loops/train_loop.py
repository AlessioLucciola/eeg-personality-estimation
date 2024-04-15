from config import SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, EPOCHS, USE_DML
from utils.utils import save_results, save_model, save_configurations
from torchmetrics import Accuracy, Recall, Precision, F1Score, AUROC
from datetime import datetime
from tqdm import tqdm
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

    # If the model is to be resumed, load the model and the optimizer
    if resume:
        data_name = PATH_MODEL_TO_RESUME
        # Start the Weights & Biases run if the configuration is set to True
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
    
    # Define the training and validation loaders based on the validation scheme
    if config["validation_scheme"] == "SPLIT":
        train_loader, val_loader = dataloaders

    training_total_step = len(train_loader) # Number of batches in the training set
    val_total_step = len(val_loader) # Number of batches in the validation set

    # Initialize the best model and the best accuracy (for saving the best model)
    best_model = None
    best_accuracy = None

    # Define the metrics
    accuracy_metric = Accuracy(task="multilabel", num_labels=len(config["labels"]))
    recall_metric = Recall(task="multilabel", num_labels=len(config["labels"]), average='macro')
    precision_metric = Precision(task="multilabel", num_labels=len(config["labels"]), average='macro')
    f1_metric = F1Score(task="multilabel", num_labels=len(config["labels"]), average='macro')
    auroc_metric = AUROC(task="multilabel", num_labels=len(config["labels"]))

    for epoch in range(RESUME_EPOCH if resume else 0, EPOCHS):

        # --Training--
        model.train() # Set the model to training mode
        # Define the tensors to store the predictions and the labels for the training set
        epoch_tr_preds = torch.tensor([]).to(device)
        epoch_tr_labels = torch.tensor([]).to(device)
        epoch_tr_outputs = torch.tensor([]).to(device)
        epoch_tr_loss = 0
        for _, tr_batch in enumerate(tqdm(train_loader, desc=f"Training Epoch [{epoch+1}/{EPOCHS}]", leave=False)):
            # Select the data and the labels
            tr_data, tr_labels = tr_batch['eeg_data'], tr_batch['labels']
            tr_data = tr_data.to(device)
            tr_labels = tr_labels.to(device)

            # Forward pass
            tr_outputs = model(tr_data) # Prediction
            epoch_tr_outputs = torch.cat((epoch_tr_outputs, tr_outputs), 0)
            
            # Loss computation
            tr_loss = criterion(tr_outputs, tr_labels)
            epoch_tr_loss += tr_loss.item()

            # Backward pass
            optimizer.zero_grad()
            tr_loss.backward()
            optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                tr_preds = (tr_outputs >= config["threshold"]).float() # Convert the predictions to binary (for each label)
                epoch_tr_preds = torch.cat((epoch_tr_preds, tr_preds), 0)
                epoch_tr_labels = torch.cat((epoch_tr_labels, tr_labels), 0)
        
        with torch.no_grad():
            if USE_DML:
                epoch_tr_preds = epoch_tr_preds.long().cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                epoch_tr_labels = epoch_tr_labels.long().cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                epoch_tr_outputs = epoch_tr_outputs.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
            
            # Compute metrics
            tr_accuracy = accuracy_metric(epoch_tr_preds, epoch_tr_labels) * 100
            tr_recall = recall_metric(epoch_tr_preds, epoch_tr_labels) * 100
            tr_precision = precision_metric(epoch_tr_preds, epoch_tr_labels) * 100
            tr_f1 = f1_metric(epoch_tr_preds, epoch_tr_labels) * 100
            tr_auroc = auroc_metric(epoch_tr_outputs, epoch_tr_labels) * 100

            print('Training -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
                .format(epoch+1, EPOCHS, epoch_tr_loss/training_total_step, tr_accuracy, tr_recall, tr_precision, tr_f1, tr_auroc))

        if config["use_wandb"]:
            wandb.log({"Training Loss": epoch_tr_loss/training_total_step})
            wandb.log({"Training Accuracy": tr_accuracy.item()})
            wandb.log({"Training Recall": tr_recall.item()})
            wandb.log({"Training Precision": tr_precision.item()})
            wandb.log({"Training F1": tr_f1.item()})
            wandb.log({"Training AUROC": tr_auroc.item()})
        
        # --Validation--
        model.eval() # Set the model to evaluation mode
        with torch.no_grad():
            # Define the tensors to store the predictions and the labels for the validation set
            epoch_val_preds = torch.tensor([]).to(device)
            epoch_val_labels = torch.tensor([]).to(device)
            epoch_val_outputs = torch.tensor([]).to(device)
            epoch_val_loss = 0

            for _, val_batch in enumerate(tqdm(val_loader, desc=f"Validation Epoch [{epoch+1}/{EPOCHS}]", leave=False)):
                # Select the data and the labels
                val_data, val_labels = val_batch['eeg_data'], val_batch['labels']
                val_data = val_data.to(device)
                val_labels = val_labels.to(device)

                # Forward pass
                val_outputs = model(val_data)
                epoch_val_outputs = torch.cat((epoch_val_outputs, val_outputs), 0)

                # Loss computation
                val_loss = criterion(val_outputs, val_labels) # Compute loss
                epoch_val_loss += val_loss.item() # Accumulate validation loss

                # Compute metrics
                val_preds = (val_outputs > config["threshold"]).long() # Convert the predictions to binary (for each label)
                epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
                epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)

            if USE_DML:
                epoch_val_preds = epoch_val_preds.long().cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                epoch_val_labels = epoch_val_labels.long().cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                epoch_val_outputs = epoch_val_outputs.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
            
            # Compute metrics
            val_accuracy = accuracy_metric(epoch_val_preds, epoch_val_labels) * 100
            val_recall = recall_metric(epoch_val_preds, epoch_val_labels) * 100
            val_precision = precision_metric(epoch_val_preds, epoch_val_labels) * 100
            val_f1 = f1_metric(epoch_val_preds, epoch_val_labels) * 100
            val_auroc = auroc_metric(epoch_val_outputs, epoch_val_labels) * 100

            if config["use_wandb"]:
                wandb.log({"Validation Loss": epoch_val_loss/val_total_step})
                wandb.log({"Validation Accuracy": val_accuracy.item()})
                wandb.log({"Validation Recall": val_recall.item()})
                wandb.log({"Validation Precision": val_precision.item()})
                wandb.log({"Validation F1": val_f1.item()})
                wandb.log({"Validation AUROC": val_auroc.item()})
            print('Validation -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
                  .format(epoch+1, EPOCHS, epoch_val_loss/val_total_step, val_accuracy, val_recall, val_precision, val_f1, val_auroc))

            if best_accuracy is None or val_accuracy < best_accuracy:
                best_accuracy = val_accuracy
                best_model = copy.deepcopy(model)
            current_results = {
                'epoch': epoch+1,
                'training_loss': epoch_tr_loss/training_total_step,
                'training_accuracy': tr_accuracy.item(),
                'training_recall': tr_recall.item(),
                'training_precision': tr_precision.item(),
                'training_f1': tr_f1.item(),
                'training_auroc': tr_auroc.item(),
                'validation_loss': epoch_val_loss/val_total_step,
                'validation_accuracy': val_accuracy.item(),
                'validation_recall': val_recall.item(),
                'validation_precision': val_precision.item(),
                'validation_f1': val_f1.item(),
                'validation_auroc': val_auroc.item()
            }
            if SAVE_RESULTS:
                save_results(data_name, current_results)
            if SAVE_MODELS:
                save_model(data_name, model, epoch)
            if epoch == EPOCHS-1 and SAVE_MODELS:
                save_model(data_name, best_model, epoch=None, is_best=True)

        #scheduler.step()