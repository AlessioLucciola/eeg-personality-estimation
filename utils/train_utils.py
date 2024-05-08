from torchmetrics import AUROC, Accuracy, Recall, Precision, F1Score
from config import LEARNING_RATE, REG, RESULTS_DIR
import torch
import json
import os

def get_optimizer(optimizer_name, parameters, lr=LEARNING_RATE, weight_decay=REG):
    if optimizer_name == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")
    
def get_scheduler(optimizer, scheduler_name, step_size, gamma):
    if scheduler_name == "StepLR":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if scheduler_name == "MultiStepLR":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=step_size, gamma=gamma)
    elif scheduler_name == "ReduceLROnPlateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=gamma, patience=step_size, verbose=True)
    else:
        raise ValueError(f"Scheduler {scheduler_name} is not supported.")
    
def get_criterion(criterion_name, smoothing_factor=0.1):
    if smoothing_factor > 0:
        print(f"--TRAIN-- Using label smoothing with epsilon = {smoothing_factor}")
    if criterion_name == "BCEWithLogitsLoss":
        # Binary Cross Entropy with Logits Loss modified to support label smoothing
        class SmoothBCEWithLogitsLoss(torch.nn.Module):
            def __init__(self, label_smoothing=0.0, reduction='mean'):
                super(SmoothBCEWithLogitsLoss, self).__init__()
                assert 0 <= label_smoothing < 1, "label_smoothing value must be between 0 and 1."
                self.label_smoothing = label_smoothing
                self.reduction = reduction
                self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction=reduction)

            def forward(self, input, target):
                if self.label_smoothing > 0:
                    positive_smoothed_labels = 1.0 - self.label_smoothing
                    negative_smoothed_labels = self.label_smoothing
                    target = target * positive_smoothed_labels + (1 - target) * negative_smoothed_labels

                loss = self.bce_with_logits(input, target)
                return loss
        return SmoothBCEWithLogitsLoss(label_smoothing=smoothing_factor)
    elif criterion_name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss(label_smoothing=smoothing_factor)
    else:
        raise ValueError(f"Criterion {criterion_name} is not supported.")

def add_dropout_to_model(model, dropout_p=0.25):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            dropout_layer = torch.nn.Dropout(p=dropout_p)
            setattr(model, name, torch.nn.Sequential(module, dropout_layer))
        elif isinstance(module, torch.nn.Module):
            # Recursively add dropout to submodules
            add_dropout_to_model(module, dropout_p=dropout_p)
    return model

def compute_average_fold_metrics(fold_metrics, fold_index):
    aggregated_fold_metrics = {
        'fold': fold_index,
        'fold_training_loss': 0,
        'fold_training_accuracy': 0,
        'fold_training_recall': 0,
        'fold_training_precision': 0,
        'fold_training_f1': 0,
        'fold_training_auroc': 0,
        'fold_validation_loss': 0,
        'fold_validation_accuracy': 0,
        'fold_validation_recall': 0,
        'fold_validation_precision': 0,
        'fold_validation_f1': 0,
        'fold_validation_auroc': 0
    }
    
    num_results = len(fold_metrics)

    # Iterate over each fold result
    for fold_results in fold_metrics:
        # Aggregate metrics
        aggregated_fold_metrics['fold_training_loss'] += fold_results['training_loss'] if fold_index != "final" else fold_results['fold_training_loss']
        aggregated_fold_metrics['fold_training_accuracy'] += fold_results['training_accuracy'] if fold_index != "final" else fold_results['fold_training_accuracy']
        aggregated_fold_metrics['fold_training_recall'] += fold_results['training_recall'] if fold_index != "final" else fold_results['fold_training_recall']
        aggregated_fold_metrics['fold_training_precision'] += fold_results['training_precision'] if fold_index != "final" else fold_results['fold_training_precision']
        aggregated_fold_metrics['fold_training_f1'] += fold_results['training_f1'] if fold_index != "final" else fold_results['fold_training_f1']
        aggregated_fold_metrics['fold_training_auroc'] += fold_results['training_auroc'] if fold_index != "final" else fold_results['fold_training_auroc']
        aggregated_fold_metrics['fold_validation_loss'] += fold_results['validation_loss'] if fold_index != "final" else fold_results['fold_validation_loss']
        aggregated_fold_metrics['fold_validation_accuracy'] += fold_results['validation_accuracy'] if fold_index != "final" else fold_results['fold_validation_accuracy']
        aggregated_fold_metrics['fold_validation_recall'] += fold_results['validation_recall'] if fold_index != "final" else fold_results['fold_validation_recall']
        aggregated_fold_metrics['fold_validation_precision'] += fold_results['validation_precision'] if fold_index != "final" else fold_results['fold_validation_precision']
        aggregated_fold_metrics['fold_validation_f1'] += fold_results['validation_f1'] if fold_index != "final" else fold_results['fold_validation_f1']
        aggregated_fold_metrics['fold_validation_auroc'] += fold_results['validation_auroc'] if fold_index != "final" else fold_results['fold_validation_auroc']

    # Calculate the average for each metric
    for key in aggregated_fold_metrics.keys():
        if key != 'fold':
            aggregated_fold_metrics[key] /= num_results
    
    return aggregated_fold_metrics

def find_best_model(dataset_name, withFold=False):
    path = RESULTS_DIR + f"/{dataset_name}/results/tr_val_results.json"

    best_accuracy = float('-inf')
    best_epoch = None
    best_fold = None

    if os.path.exists(path):
        with open(path, 'r') as json_file:
            results = json.load(json_file)

            for result in results:
                if withFold:
                    fold = result['fold']
                epoch = result['epoch']
                accuracy = result['validation_accuracy']

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_epoch = epoch
                    if withFold:
                        best_fold = fold
    
    config_path = RESULTS_DIR + f"/{dataset_name}/configurations.json"
    if os.path.exists(config_path):
        with open(config_path, 'r') as json_file:
            configurations = json.load(json_file)
            configurations['best_accuracy'] = best_accuracy
            configurations['best_epoch'] = best_epoch
            if withFold:
                configurations['best_fold'] = best_fold
        with open(config_path, 'w') as json_file:
            json.dump(configurations, json_file, indent=2)

# Function to reset the weights of a model
def reset_weights(model, weights):
  print("--TRAIN-- Resetting the weights of the model at epoch 1..")
  model.load_state_dict(weights)
  return model

def load_metrics(num_labels):
    accuracy_metric = Accuracy(task="multilabel", num_labels=num_labels)
    recall_metric = Recall(task="multilabel", num_labels=num_labels, average='macro')
    precision_metric = Precision(task="multilabel", num_labels=num_labels, average='macro')
    f1_metric = F1Score(task="multilabel", num_labels=num_labels, average='macro')
    auroc_metric = AUROC(task="multilabel", num_labels=num_labels)
    label_metrics = {
        label: {
            'accuracy': Accuracy(task="binary"),
            'recall': Recall(task="binary", average='macro'),
            'precision': Precision(task="binary", average='macro'),
            'f1': F1Score(task="binary", average='macro'),
            'auroc': AUROC(task="binary")
        } for label in range(num_labels)
    }
    return accuracy_metric, recall_metric, precision_metric, f1_metric, auroc_metric, label_metrics

def measure_performances(
        acc_metric: Accuracy,
        rec_metric: Recall,
        prec_metric: Precision,
        f1_metric: F1Score,
        auroc_metric: AUROC,
        preds: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor
    ):
    accuracy = acc_metric(preds, labels) * 100
    recall = rec_metric(preds, labels) * 100
    precision = prec_metric(preds, labels) * 100
    f1 = f1_metric(preds, labels) * 100
    auroc = auroc_metric(outputs, labels) * 100
    return accuracy, recall, precision, f1, auroc
