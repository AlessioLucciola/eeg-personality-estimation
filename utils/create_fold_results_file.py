from config import RESULTS_DIR
import numpy as np
import json
import os

def upload_results_file(model_folder):
    model_path = os.path.join(RESULTS_DIR, model_folder)
    configurations = None
    general_results = None
    # Load model configurations
    with open(f'{model_path}/configurations.json', 'r') as f:
        configurations = json.load(f)
    # Load general results
    with open(f'{model_path}/results/tr_val_results.json', 'r') as f:
        general_results = json.load(f)
    return configurations, general_results

def compute_fold_results(configurations, general_results, metric="validation_accuracy"):
    folds = set([f["fold"] for f in general_results if "fold" in f])
    final_metrics = []
    avg_metrics = {}
    for fold in folds:
        fold_results = [f for f in general_results if f["fold"] == fold]
        index_fold_best_results = fold_results.index(max(fold_results, key=lambda x: x[metric]))
        best_results = fold_results[index_fold_best_results]
        final_fold_metrics = {}
        final_fold_metrics["fold"] = fold
        final_fold_metrics["fold_training_loss"] = best_results["training_loss"]
        final_fold_metrics["fold_validation_loss"] = best_results["validation_loss"]
        final_fold_metrics["fold_training_accuracy"] = best_results["training_accuracy"]
        final_fold_metrics["fold_validation_accuracy"] = best_results["validation_accuracy"]
        final_fold_metrics["fold_training_f1"] = best_results["training_f1"]
        final_fold_metrics["fold_validation_f1"] = best_results["validation_f1"]
        final_fold_metrics["fold_training_precision"] = best_results["training_precision"]
        final_fold_metrics["fold_validation_precision"] = best_results["validation_precision"]
        final_fold_metrics["fold_training_recall"] = best_results["training_recall"]
        final_fold_metrics["fold_validation_recall"] = best_results["validation_recall"]
        final_fold_metrics["fold_training_auroc"] = best_results["training_auroc"]
        final_fold_metrics["fold_validation_auroc"] = best_results["validation_auroc"]
        if configurations["evaluate_each_label"]:
            for label in range(len(configurations["labels"].keys())):
                final_fold_metrics[f"fold_training_f1_label_{label}"] = best_results[f"training_f1_label_{label}"]
                final_fold_metrics[f"fold_validation_f1_label_{label}"] = best_results[f"validation_f1_label_{label}"]
                final_fold_metrics[f"fold_training_precision_label_{label}"] = best_results[f"training_precision_label_{label}"]
                final_fold_metrics[f"fold_validation_precision_label_{label}"] = best_results[f"validation_precision_label_{label}"]
                final_fold_metrics[f"fold_training_recall_label_{label}"] = best_results[f"training_recall_label_{label}"]
                final_fold_metrics[f"fold_validation_recall_label_{label}"] = best_results[f"validation_recall_label_{label}"]
                final_fold_metrics[f"fold_training_auroc_label_{label}"] = best_results[f"training_auroc_label_{label}"]
                final_fold_metrics[f"fold_validation_auroc_label_{label}"] = best_results[f"validation_auroc_label_{label}"]
                final_fold_metrics[f"fold_training_accuracy_label_{label}"] = best_results[f"training_accuracy_label_{label}"]
                final_fold_metrics[f"fold_validation_accuracy_label_{label}"] = best_results[f"validation_accuracy_label_{label}"]
        final_metrics.append(final_fold_metrics)
    avg_metrics["fold"] = "final"
    for key in ["fold_training_loss", "fold_validation_loss", "fold_training_accuracy", "fold_validation_accuracy",
                "fold_training_f1", "fold_validation_f1", "fold_training_precision", "fold_validation_precision",
                "fold_training_recall", "fold_validation_recall", "fold_training_auroc", "fold_validation_auroc"]:
        values = [f[key] for f in final_metrics]
        avg_metrics[key] = f"{np.mean(values):.2f} +- {np.std(values):.2f}"
    
    if configurations["evaluate_each_label"]:
        for label in range(len(configurations["labels"].keys())):
            for key_prefix in ["fold_training_f1_label_", "fold_validation_f1_label_", "fold_training_precision_label_", 
                               "fold_validation_precision_label_", "fold_training_recall_label_", "fold_validation_recall_label_", 
                               "fold_training_auroc_label_", "fold_validation_auroc_label_", "fold_training_accuracy_label_", 
                               "fold_validation_accuracy_label_"]:
                key = f"{key_prefix}{label}"
                values = [f[key] for f in final_metrics]
                avg_metrics[key] = f"{np.mean(values):.2f} +- {np.std(values):.2f}"

    final_metrics.append(avg_metrics)
    return final_metrics

def save_metrics(final_results, model_folder):
    model_path = os.path.join(RESULTS_DIR, model_folder)
    with open(f'{model_path}/results/fold_results.json', 'w') as f:
        json.dump(final_results, f, indent=4)            

if __name__ == "__main__":
    model_folder = "ViT_2024-07-20_20-50-17"
    configurations, general_results = upload_results_file(model_folder=model_folder)
    final_results = compute_fold_results(configurations, general_results, metric="validation_f1")
    save_metrics(final_results, model_folder)