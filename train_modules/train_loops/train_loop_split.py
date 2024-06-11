from config import SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, EPOCHS, USE_DML
from utils.train_utils import find_best_model, load_metrics, measure_performances
from utils.utils import save_results, save_model, save_configurations
from datetime import datetime
from tqdm import tqdm
import torch
import wandb

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
    
    # Define the training and validation loaders
    train_loader, val_loader = dataloaders
    # Define the criterion to used depending on the use of the triplet loss
    if config["use_triplet"]:
        triplet_criterion = criterion[0]
        binary_criterion = criterion[1]
    else:
        binary_criterion = criterion

    training_total_step = len(train_loader) # Number of batches in the training set
    val_total_step = len(val_loader) # Number of batches in the validation set

    # Define the metrics
    accuracy_metric, recall_metric, precision_metric, f1_metric, auroc_metric, label_metrics = load_metrics(num_labels=len(config["labels"]))

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
            if config["use_triplet"]:
                anchor_sample, positive_sample, negative_sample = tr_batch[0], tr_batch[1], tr_batch[2]
                tr_data, tr_labels = anchor_sample['spectrogram'], anchor_sample['labels']
                tr_positive_data, tr_positive_labels = positive_sample['spectrogram'], positive_sample['labels']
                tr_negative_data, tr_negative_labels = negative_sample['spectrogram'], negative_sample['labels']
                tr_data = tr_data.to(device)
                tr_positive_data = tr_positive_data.to(device)
                tr_negative_data = tr_negative_data.to(device)
                tr_labels = tr_labels.to(device)
                tr_positive_labels = tr_positive_labels.to(device)
                tr_negative_labels = tr_negative_labels.to(device)
                
                tr_outputs = model(tr_data) # Prediction for the anchor sample
                epoch_tr_outputs = torch.cat((epoch_tr_outputs, tr_outputs), 0) # Concatenate the predictions for the anchor sample
                tr_positive_outputs = model(tr_positive_data) # Prediction for the positive sample
                tr_negative_outputs = model(tr_negative_data) # Prediction for the negative sample

                tr_triplet_loss = triplet_criterion(tr_outputs, tr_positive_outputs, tr_negative_outputs) # Compute the triplet loss
                tr_loss_anchor = binary_criterion(tr_outputs, tr_labels) # Compute the loss for the anchor sample
                tr_loss_positive = binary_criterion(tr_positive_outputs, tr_positive_labels) # Compute the loss for the positive sample
                tr_loss_negative = binary_criterion(tr_negative_outputs, tr_negative_labels) # Compute the loss for the negative sample
                tr_loss = tr_triplet_loss + tr_loss_anchor + tr_loss_positive + tr_loss_negative # Compute the total loss
            else:
                tr_data, tr_labels = tr_batch['spectrogram'], tr_batch['labels']
                tr_data = tr_data.to(device)
                tr_labels = tr_labels.to(device)

                # Forward pass
                tr_outputs = model(tr_data) # Prediction
                epoch_tr_outputs = torch.cat((epoch_tr_outputs, tr_outputs), 0)
                
                # Loss computation
                tr_loss = criterion(tr_outputs, tr_labels)

            epoch_tr_loss += tr_loss.item() # Accumulate training loss

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
            epoch_tr_preds = epoch_tr_preds.long()
            epoch_tr_labels = epoch_tr_labels.long()
            epoch_tr_outputs = epoch_tr_outputs.float()
            if USE_DML:
                epoch_tr_preds = epoch_tr_preds.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                epoch_tr_labels = epoch_tr_labels.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                epoch_tr_outputs = epoch_tr_outputs.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
            
            # Compute general metrics
            tr_accuracy, tr_recall, tr_precision, tr_f1, tr_auroc = measure_performances(
                acc_metric=accuracy_metric,
                rec_metric=recall_metric,
                prec_metric=precision_metric,
                f1_metric=f1_metric,
                auroc_metric=auroc_metric,
                preds=epoch_tr_preds,
                labels=epoch_tr_labels,
                outputs=epoch_tr_outputs
            )

            print('Training -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
                .format(epoch+1, EPOCHS, epoch_tr_loss/training_total_step, tr_accuracy, tr_recall, tr_precision, tr_f1, tr_auroc))

            # Compute metrics for each label if the associated parameter is True
            if config["evaluate_each_label"]:
                for label, metrics in label_metrics.items():
                    label_accuracy, label_recall, label_precision, label_f1, label_auroc = measure_performances(
                        acc_metric=metrics['accuracy'],
                        rec_metric=metrics['recall'],
                        prec_metric=metrics['precision'],
                        f1_metric=metrics['f1'],
                        auroc_metric=metrics['auroc'],
                        preds=epoch_tr_preds[:, label],
                        labels=epoch_tr_labels[:, label],
                        outputs=epoch_tr_outputs[:, label]
                    )
                    print(f'Training -> Epoch [{epoch+1}/{EPOCHS}], Metrics for Label {label} -> Accuracy: {label_accuracy}, Recall: {label_recall}, Precision: {label_precision}, F1: {label_f1}, AUROC: {label_auroc}')

        if config["use_wandb"]:
            wandb.log({"Training Loss": epoch_tr_loss/training_total_step})
            wandb.log({"Training Accuracy": tr_accuracy.item()})
            wandb.log({"Training Recall": tr_recall.item()})
            wandb.log({"Training Precision": tr_precision.item()})
            wandb.log({"Training F1": tr_f1.item()})
            wandb.log({"Training AUROC": tr_auroc.item()})
            if config["evaluate_each_label"]:
                for label, metrics in label_metrics.items():
                    wandb.log({f"Training Accuracy Label {label}": label_accuracy.item()})
                    wandb.log({f"Training Recall Label {label}": label_recall.item()})
                    wandb.log({f"Training Precision Label {label}": label_precision.item()})
                    wandb.log({f"Training F1 Label {label}": label_f1.item()})
                    wandb.log({f"Training AUROC Label {label}": label_auroc.item()})
        
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
                val_data, val_labels = val_batch['spectrogram'], val_batch['labels']
                val_data = val_data.to(device)
                val_labels = val_labels.to(device)

                # Forward pass
                val_outputs = model(val_data)
                epoch_val_outputs = torch.cat((epoch_val_outputs, val_outputs), 0)

                # Loss computation
                val_loss = binary_criterion(val_outputs, val_labels) # Compute loss
                epoch_val_loss += val_loss.item() # Accumulate validation loss

                # Compute metrics
                val_preds = (val_outputs >= config["threshold"]).long() # Convert the predictions to binary (for each label)
                epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
                epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)

            epoch_val_preds = epoch_val_preds.long()
            epoch_val_labels = epoch_val_labels.long()
            epoch_val_outputs = epoch_val_outputs.float()
            if USE_DML:
                epoch_val_preds = epoch_val_preds.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                epoch_val_labels = epoch_val_labels.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                epoch_val_outputs = epoch_val_outputs.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
            
            # Compute metrics
            val_accuracy, val_recall, val_precision, val_f1, val_auroc = measure_performances(
                acc_metric=accuracy_metric,
                rec_metric=recall_metric,
                prec_metric=precision_metric,
                f1_metric=f1_metric,
                auroc_metric=auroc_metric,
                preds=epoch_val_preds,
                labels=epoch_val_labels,
                outputs=epoch_val_outputs
            )

            print('Validation -> Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
                  .format(epoch+1, EPOCHS, epoch_val_loss/val_total_step, val_accuracy, val_recall, val_precision, val_f1, val_auroc))

            # Compute metrics for each label if the associated parameter is True
            if config["evaluate_each_label"]:
                for label, metrics in label_metrics.items():
                    label_accuracy, label_recall, label_precision, label_f1, label_auroc = measure_performances(
                        acc_metric=metrics['accuracy'],
                        rec_metric=metrics['recall'],
                        prec_metric=metrics['precision'],
                        f1_metric=metrics['f1'],
                        auroc_metric=metrics['auroc'],
                        preds=epoch_val_preds[:, label],
                        labels=epoch_val_labels[:, label],
                        outputs=epoch_val_outputs[:, label]
                    )
                    print(f'Validation -> Epoch [{epoch+1}/{EPOCHS}], Metrics for Label {label} -> Accuracy: {label_accuracy}, Recall: {label_recall}, Precision: {label_precision}, F1: {label_f1}, AUROC: {label_auroc}')

            if config["use_wandb"]:
                wandb.log({"Validation Loss": epoch_val_loss/val_total_step})
                wandb.log({"Validation Accuracy": val_accuracy.item()})
                wandb.log({"Validation Recall": val_recall.item()})
                wandb.log({"Validation Precision": val_precision.item()})
                wandb.log({"Validation F1": val_f1.item()})
                wandb.log({"Validation AUROC": val_auroc.item()})
                if config["evaluate_each_label"]:
                    for label, metrics in label_metrics.items():
                        wandb.log({f"Validation Accuracy Label {label}": label_accuracy.item()})
                        wandb.log({f"Validation Recall Label {label}": label_recall.item()})
                        wandb.log({f"Validation Precision Label {label}": label_precision.item()})
                        wandb.log({f"Validation F1 Label {label}": label_f1.item()})
                        wandb.log({f"Validation AUROC Label {label}": label_auroc.item()})

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
            if config["evaluate_each_label"]:
                for label, metrics in label_metrics.items():
                    current_results[f'training_accuracy_label_{label}'] = label_accuracy.item()
                    current_results[f'training_recall_label_{label}'] = label_recall.item()
                    current_results[f'training_precision_label_{label}'] = label_precision.item()
                    current_results[f'training_f1_label_{label}'] = label_f1.item()
                    current_results[f'training_auroc_label_{label}'] = label_auroc.item()
                    current_results[f'validation_accuracy_label_{label}'] = label_accuracy.item()
                    current_results[f'validation_recall_label_{label}'] = label_recall.item()
                    current_results[f'validation_precision_label_{label}'] = label_precision.item()
                    current_results[f'validation_f1_label_{label}'] = label_f1.item()
                    current_results[f'validation_auroc_label_{label}'] = label_auroc.item()
            if SAVE_RESULTS:
                save_results(data_name, current_results)
            
            find_best_model(data_name, withFold=False) # Find the best model based on the validation accuracy and save the associated epoch and fold in the configuration file

            if SAVE_MODELS:
                save_model(data_name, model, epoch)

        #scheduler.step()