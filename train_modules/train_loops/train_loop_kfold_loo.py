from utils.utils import resume_starting_weights, save_results, save_model, save_configurations, save_fold_results, resume_folds_metrics, save_starting_weights
from utils.train_utils import compute_average_fold_metrics, find_best_model, load_metrics, measure_performances, reset_weights
from config import SAVE_MODELS, SAVE_RESULTS, PATH_MODEL_TO_RESUME, RESUME_EPOCH, EPOCHS, USE_DML, RESUME_FOLD
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

    starting_weights = None # Variable to store the starting weights of the model (useful for resetting the model at each fold)
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
        starting_weights = resume_starting_weights(data_name) # Load the starting weights of the model
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
        starting_weights = copy.deepcopy(model.state_dict()) # Save the starting weights of the model
        save_starting_weights(weights=starting_weights, path=data_name) # Save the starting weights of the model
        
    if resume:
        folds_metrics = resume_folds_metrics(data_name) # List to store the metrics in all folds
    else:
        folds_metrics = []
    
    dataloaders_num = len(dataloaders) # Number of dataloaders (folds) - k in k-fold CV, number of subjects in LOOCV
    for fold_i, (train_loader, val_loader) in list(dataloaders.items()):
        if config["validation_scheme"] == "LOOCV":
            i, subject_id = fold_i # i is the index of the fold, subject_id is the subject to leave out for validation in LOOCV
        else:
            i = fold_i # i is the index of the fold
        # If the training is to be resumed, skip the folds until the one to resume
        if resume and i < RESUME_FOLD-1:
            continue
        training_total_step = len(train_loader) # Number of batches in the training set
        val_total_step = len(val_loader) # Number of batches in the validation set

        if resume:
            fold_metrics = resume_folds_metrics(data_name, fold=RESUME_FOLD, epoch=RESUME_EPOCH) # List to store the metrics in a fold
        else:
            fold_metrics = [] # List to store the metrics in a fold

        # Define the metrics
        accuracy_metric, recall_metric, precision_metric, f1_metric, auroc_metric, label_metrics = load_metrics(num_labels=len(config["labels"]))

        fold_model = model # Copy the model for each fold
        fold_model = reset_weights(model=fold_model, weights=starting_weights) # Reset the weights of the model for each fold
        fold_optimizer = optimizer # Copy the optimizer for each fold

        for epoch in range(RESUME_EPOCH if resume else 0, EPOCHS):

            # --Training--
            fold_model.train() # Set the model to training mode
            # Define the tensors to store the predictions and the labels for the training set
            epoch_tr_preds = torch.tensor([]).to(device)
            epoch_tr_labels = torch.tensor([]).to(device)
            epoch_tr_outputs = torch.tensor([]).to(device)
            epoch_tr_loss = 0
            training_description = f"Training Fold [{i+1}/{dataloaders_num}], Epoch [{epoch+1}/{EPOCHS}]"
            if config["validation_scheme"] == "LOOCV":
                training_description += f", Subject [{subject_id}]"
            for _, tr_batch in enumerate(tqdm(train_loader, desc=training_description, leave=False)):
                # Select the data and the labels
                tr_data, tr_labels = tr_batch['spectrogram'], tr_batch['labels']
                tr_data = tr_data.to(device)
                tr_labels = tr_labels.to(device)

                # Forward pass
                tr_outputs = fold_model(tr_data) # Prediction
                epoch_tr_outputs = torch.cat((epoch_tr_outputs, tr_outputs), 0)
                
                # Loss computation
                tr_loss = criterion(tr_outputs, tr_labels)
                epoch_tr_loss += tr_loss.item()

                # Backward pass
                fold_optimizer.zero_grad()
                tr_loss.backward()
                fold_optimizer.step()
                
                # Compute metrics
                with torch.no_grad():
                    tr_preds = (tr_outputs >= config["threshold"]) # Convert the predictions to binary (for each label)
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

                print('Training -> Fold [{}/{}], Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
                    .format(i+1, dataloaders_num, epoch+1, EPOCHS, epoch_tr_loss/training_total_step, tr_accuracy, tr_recall, tr_precision, tr_f1, tr_auroc))
                
                labels_tr_performance = {}
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
                        labels_tr_performance[label] = {
                            'accuracy': label_accuracy,
                            'recall': label_recall,
                            'precision': label_precision,
                            'f1': label_f1,
                            'auroc': label_auroc
                        }
                        print(f'Training -> Fold [{i+1}/{dataloaders_num}], Epoch [{epoch+1}/{EPOCHS}], Metrics for Label {label} -> Accuracy: {label_accuracy}, Recall: {label_recall}, Precision: {label_precision}, F1: {label_f1}, AUROC: {label_auroc}')

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
            fold_model.eval() # Set the model to evaluation mode
            with torch.no_grad():
                # Define the tensors to store the predictions and the labels for the validation set
                epoch_val_preds = torch.tensor([]).to(device)
                epoch_val_labels = torch.tensor([]).to(device)
                epoch_val_outputs = torch.tensor([]).to(device)
                epoch_val_loss = 0

                validation_description = f"Validation Fold [{i+1}/{dataloaders_num}], Epoch [{epoch+1}/{EPOCHS}]"
                if config["validation_scheme"] == "LOOCV":
                    validation_description += f", Subject [{subject_id}]"
                for _, val_batch in enumerate(tqdm(val_loader, desc=validation_description, leave=False)):
                    # Select the data and the labels
                    val_data, val_labels = val_batch['spectrogram'], val_batch['labels']
                    val_data = val_data.to(device)
                    val_labels = val_labels.to(device)

                    # Forward pass
                    val_outputs = fold_model(val_data)
                    epoch_val_outputs = torch.cat((epoch_val_outputs, val_outputs), 0)

                    # Loss computation
                    val_loss = criterion(val_outputs, val_labels) # Compute loss
                    epoch_val_loss += val_loss.item() # Accumulate validation loss

                    # Compute metrics
                    val_preds = (val_outputs > config["threshold"]) # Convert the predictions to binary (for each label)
                    epoch_val_preds = torch.cat((epoch_val_preds, val_preds), 0)
                    epoch_val_labels = torch.cat((epoch_val_labels, val_labels), 0)

                epoch_val_preds = epoch_val_preds.long()
                epoch_val_labels = epoch_val_labels.long()
                epoch_val_outputs = epoch_val_outputs.float()
                if USE_DML:
                    epoch_val_preds = epoch_val_preds.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                    epoch_val_labels = epoch_val_labels.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                    epoch_val_outputs = epoch_val_outputs.cpu() # Convert to CPU to avoid DirectML errors (only for DirectML)
                
                # Compute general metrics
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

                print('Validation -> Fold [{}/{}], Epoch [{}/{}], Loss: {:.4f}, Accuracy: {:.4f}%, Recall: {:.4f}%, Precision: {:.4f}%, F1: {:.4f}%, AUROC: {:.4f}%'
                    .format(i+1, dataloaders_num, epoch+1, EPOCHS, epoch_val_loss/val_total_step, val_accuracy, val_recall, val_precision, val_f1, val_auroc))
                
                labels_val_performance = {}
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
                        labels_val_performance[label] = {
                            'accuracy': label_accuracy,
                            'recall': label_recall,
                            'precision': label_precision,
                            'f1': label_f1,
                            'auroc': label_auroc
                        }
                        print(f'Validation -> Fold [{i+1}/{dataloaders_num}], Epoch [{epoch+1}/{EPOCHS}], Metrics for Label {label} -> Accuracy: {label_accuracy}, Recall: {label_recall}, Precision: {label_precision}, F1: {label_f1}, AUROC: {label_auroc}')

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

            current_fold_epoch_results = {
                'fold': i+1,
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
            if config["validation_scheme"] == "LOOCV":
                current_fold_epoch_results["subject"] = subject_id
            if config["evaluate_each_label"]:
                for label, metrics in label_metrics.items():
                    current_fold_epoch_results[f'training_accuracy_label_{label}'] = labels_tr_performance[label]['accuracy'].item()
                    current_fold_epoch_results[f'training_recall_label_{label}'] = labels_tr_performance[label]['recall'].item()
                    current_fold_epoch_results[f'training_precision_label_{label}'] = labels_tr_performance[label]['precision'].item()
                    current_fold_epoch_results[f'training_f1_label_{label}'] = labels_tr_performance[label]['f1'].item()
                    current_fold_epoch_results[f'training_auroc_label_{label}'] = labels_tr_performance[label]['auroc'].item()
                    current_fold_epoch_results[f'validation_accuracy_label_{label}'] = labels_val_performance[label]['accuracy'].item()
                    current_fold_epoch_results[f'validation_recall_label_{label}'] = labels_val_performance[label]['recall'].item()
                    current_fold_epoch_results[f'validation_precision_label_{label}'] = labels_val_performance[label]['precision'].item()
                    current_fold_epoch_results[f'validation_f1_label_{label}'] = labels_val_performance[label]['f1'].item()
                    current_fold_epoch_results[f'validation_auroc_label_{label}'] = labels_val_performance[label]['auroc'].item()
            fold_metrics.append(current_fold_epoch_results) # Append the results for the current fold and epoch

            if SAVE_RESULTS:
                save_results(data_name, current_fold_epoch_results)
            
            find_best_model(data_name, withFold=True) # Find the best model based on the validation accuracy and save the associated epoch and fold in the configuration file

            if SAVE_MODELS:
                save_model(data_name, model=fold_model, epoch=epoch+1, fold=i+1, is_best=False)

            #scheduler.step() # Update the learning rate
        
        final_fold_metrics = compute_average_fold_metrics(fold_metrics=fold_metrics, fold_index=i+1, evaluate_each_label=config['evaluate_each_label'], num_labels=len(config['labels'])) # Compute the average metrics for the current fold
        folds_metrics.append(final_fold_metrics) # Append the results for the current fold
        # At the end of each fold, save the results if the configuration is set to True
        if SAVE_RESULTS:
            save_fold_results(data_name, final_fold_metrics)
    
    aggregated_fold_metrics = compute_average_fold_metrics(fold_metrics=folds_metrics, fold_index="final", evaluate_each_label=config['evaluate_each_label'], num_labels=len(config['labels'])) # Compute the average metrics across all folds
    # At the end of the training, save the aggregated results of the folds if the configuration is set to True
    if SAVE_RESULTS:
        save_fold_results(data_name, aggregated_fold_metrics)