from torchmetrics import AUROC, Accuracy, Recall, Precision, F1Score
from config import LEARNING_RATE, REG, RESULTS_DIR, USE_DML
from torchprofile import profile_macs
import torch
import math
import json
import os

def get_optimizer(optimizer_name, parameters, lr=LEARNING_RATE, weight_decay=REG):
    if optimizer_name == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "SGD":
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
    elif scheduler_name == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=step_size, eta_min=5e-5)
    else:
        raise ValueError(f"Scheduler {scheduler_name} is not supported.")
    
def get_criterion(criterion_name, smoothing_factor=0.1):
    if smoothing_factor > 0:
        print(f"--TRAIN-- Using label smoothing with epsilon = {smoothing_factor}")
    if criterion_name == "BCEWithLogitsLoss":
        # Binary Cross Entropy with Logits Loss modified to support label smoothing
        return load_BCEWithLogitsLoss(smoothing_factor=smoothing_factor)
    elif criterion_name == "CrossEntropyLoss":
        return torch.nn.CrossEntropyLoss(label_smoothing=smoothing_factor)
    elif criterion_name == "TripletMarginLoss":
        return tuple((torch.nn.TripletMarginLoss(margin=1.0, p=2.0, reduction='mean'), load_BCEWithLogitsLoss(smoothing_factor=smoothing_factor)))
    else:
        raise ValueError(f"Criterion {criterion_name} is not supported.")

def load_BCEWithLogitsLoss(smoothing_factor=0.1):
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

def add_dropout_to_model(model, dropout_p=0.25):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.ReLU):
            dropout_layer = torch.nn.Dropout(p=dropout_p)
            setattr(model, name, torch.nn.Sequential(module, dropout_layer))
        elif isinstance(module, torch.nn.Module):
            # Recursively add dropout to submodules
            add_dropout_to_model(module, dropout_p=dropout_p)
    return model

def compute_average_fold_metrics(fold_metrics, fold_index, evaluate_each_label=False, num_labels=4):
    metrics = ['loss', 'accuracy', 'recall', 'precision', 'f1', 'auroc'] # Define the metrics to aggregate
    final_fold_metrics = {
        'fold': fold_index,
    }
    # Initialize the general metrics (considering all labels)
    for metric in metrics:
        final_fold_metrics[f'fold_training_{metric}'] = 0
        final_fold_metrics[f'fold_validation_{metric}'] = 0

    # Initialize the metrics for each label if the evaluation is done for each label
    if evaluate_each_label:
        for label in range(num_labels):
            for metric in metrics:
                if metric != 'loss':
                    final_fold_metrics[f'fold_training_{metric}_label_{label}'] = 0
                    final_fold_metrics[f'fold_validation_{metric}_label_{label}'] = 0
    
    # If the fold is final, aggregate the metrics of each fold
    if fold_index == "final":
        num_results = len(fold_metrics) # Get the number of results

        # Iterate over each fold result
        for fold_results in fold_metrics:
            # Aggregate metrics
            for metric in metrics:
                #final_fold_metrics[f'fold_training_{metric}'] += fold_results[f'training_{metric}'] if fold_index != "final" else fold_results[f'fold_training_{metric}']
                #final_fold_metrics[f'fold_validation_{metric}'] += fold_results[f'validation_{metric}'] if fold_index != "final" else fold_results[f'fold_validation_{metric}']
                final_fold_metrics[f'fold_training_{metric}'] += fold_results[f'fold_training_{metric}']
                final_fold_metrics[f'fold_validation_{metric}'] += fold_results[f'fold_validation_{metric}']
                if evaluate_each_label:
                    for label in range(num_labels):
                        if metric != 'loss':
                            #final_fold_metrics[f'fold_training_{metric}_label_{label}'] += fold_results[f'training_{metric}_label_{label}'] if fold_index != "final" else fold_results[f'fold_training_{metric}_label_{label}']
                            #final_fold_metrics[f'fold_validation_{metric}_label_{label}'] += fold_results[f'validation_{metric}_label_{label}'] if fold_index != "final" else fold_results[f'fold_validation_{metric}_label_{label}']
                            final_fold_metrics[f'fold_training_{metric}_label_{label}'] += fold_results[f'fold_training_{metric}_label_{label}']
                            final_fold_metrics[f'fold_validation_{metric}_label_{label}'] += fold_results[f'fold_validation_{metric}_label_{label}']

        # Calculate the average for each metric
        for key in final_fold_metrics.keys():
            if key != 'fold':
                final_fold_metrics[key] /= num_results
        
        return final_fold_metrics
    else:
        # If the fold is not final, simply return the metrics of the fold with the highest validation accuracy
        max_accuracy = float('-inf')
        max_accuracy_epoch = None
        for fi, fold in enumerate(fold_metrics):
            if fold['validation_accuracy'] > max_accuracy:
                max_accuracy = fold['validation_accuracy']
                max_accuracy_epoch = fi

        fold_results = fold_metrics[max_accuracy_epoch]
        # Aggregate metrics
        for metric in metrics:
            final_fold_metrics[f'fold_training_{metric}'] += fold_results[f'training_{metric}']
            final_fold_metrics[f'fold_validation_{metric}'] += fold_results[f'validation_{metric}']
            if evaluate_each_label and metric != 'loss':
                for label in range(num_labels):
                    final_fold_metrics[f'fold_training_{metric}_label_{label}'] = fold_results[f'training_{metric}_label_{label}']
                    final_fold_metrics[f'fold_validation_{metric}_label_{label}'] = fold_results[f'validation_{metric}_label_{label}']
        return final_fold_metrics

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

def load_metrics(num_labels, device):
    if USE_DML:
        device = torch.device("cpu") # Compatability with DirectML
    accuracy_metric = Accuracy(task="multilabel", num_labels=num_labels, average='micro').to(device)
    recall_metric = Recall(task="multilabel", num_labels=num_labels, average='micro').to(device)
    precision_metric = Precision(task="multilabel", num_labels=num_labels, average='micro').to(device)
    f1_metric = F1Score(task="multilabel", num_labels=num_labels, average='micro').to(device)
    auroc_metric = AUROC(task="multilabel", num_labels=num_labels).to(device)
    label_metrics = {
        label: {
            'accuracy': Accuracy(task="binary").to(device),
            'recall': Recall(task="binary", average='micro').to(device),
            'precision': Precision(task="binary", average='micro').to(device),
            'f1': F1Score(task="binary", average='micro').to(device),
            'auroc': AUROC(task="binary", average="micro").to(device)
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

def get_positional_encoding(positional_encoding_name, max_position_embeddings=1024, hidden_size=None):
    if positional_encoding_name == "sinusoidal":
        class GetSinusoidalPositionalEmbeddings(torch.nn.Module):
            def __init__(
                    self,
                    max_position_embeddings: int = max_position_embeddings
            ):
                super().__init__()
                self.max_position_embeddings = max_position_embeddings

            def forward(self, x):
                sequence_length, embeddings_dim = self.max_position_embeddings, x.shape[-1]
                pe = torch.zeros(sequence_length, embeddings_dim, device=x.device)
                position = torch.arange(0, sequence_length, dtype=torch.float, device=x.device).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, embeddings_dim, 2, dtype=torch.float, device=x.device) * 
                                    -(math.log(10000.0) / embeddings_dim))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).repeat(x.shape[0], 1, 1)
                pe = pe[:, :x.shape[1], :]  # Ensure the positional encoding matches the input sequence length
                assert pe.shape == x.shape, f"Expected shape {x.shape}, but got {pe.shape}"
                return pe
        return GetSinusoidalPositionalEmbeddings(max_position_embeddings=max_position_embeddings)
    elif positional_encoding_name == "learnable":
        class GetLearnablePositionalEmbeddings(torch.nn.Module):
            def __init__(
                    self,
                    hidden_size: int,
                    max_position_embeddings: int = max_position_embeddings,
            ):
                super().__init__()
                self.max_position_embeddings = max_position_embeddings
                self.hidden_size = hidden_size
                self.embedder = torch.nn.Embedding(self.max_position_embeddings, self.hidden_size)

            def forward(self, x):
                pe = self.embedder(torch.arange(x.shape[1], device=x.device)).repeat(x.shape[0], 1, 1)
                assert pe.shape == x.shape, f"Expected shape {x.shape}, but got {pe.shape}"
                return pe
        return GetLearnablePositionalEmbeddings(max_position_embeddings=max_position_embeddings, hidden_size=hidden_size)
    else:
        raise ValueError(f"Positional encoding {positional_encoding_name} is not supported.")
    
def get_model_params(model):
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model_params

def count_model_MACs(model, input):
    model.eval()
    macs = profile_macs(model, input)
    return macs
