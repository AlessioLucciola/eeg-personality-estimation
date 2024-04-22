import os
import json
import matplotlib.pyplot as plt
from config import *

def read_train_val_results(result_folder, validation_scheme=None):
    train_validation_results = None
    fold_results = None

    script_directory = os.path.dirname(os.path.realpath(__file__))
    test_file_name = os.path.join(script_directory, '..', 'results', result_folder, 'results', 'tr_val_results.json')
    if os.path.exists(test_file_name):
        with open(test_file_name, 'r') as file:
            try:
                train_validation_results = json.load(file)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {test_file_name}: {e}")
    else:
        raise ValueError(f"Traain and validation results for {result_folder} don't exist")
    
    if validation_scheme == "K_FOLDCV" or validation_scheme == "LOOCV":
        fold_results_file_name = os.path.join(script_directory, '..', 'results', result_folder, 'results', 'fold_results.json')
        if os.path.exists(fold_results_file_name):
            with open(fold_results_file_name, 'r') as file:
                try:
                    fold_results = json.load(file)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON in file {test_file_name}: {e}")
        else:
            raise ValueError(f"Test results for {result_folder} don't exist")

    return train_validation_results, fold_results

def create_train_val_line_plots(metrics, data, models_name, validation_schemes, validation_settings, save_plot_prefix="plot"):
    script_directory = os.path.dirname(__file__)
    for _, metric in enumerate(metrics):
        if (("LOOCV" in validation_schemes or "K-FOLDCV" in validation_schemes) and "SPLIT" in validation_schemes):
            _, ax = plt.subplots(figsize=(10, 7))
            lines = []
            labels = []
            for j, model_data in enumerate(data):
                train_label = f"{models_name[j]} Train {metric[1]}"
                val_label = f"{models_name[j]} Validation {metric[1]}"
                if validation_schemes[j] == "SPLIT":
                    train_values = [epoch[f'training_{metric[0]}'] for epoch in model_data]
                    val_values = [epoch[f'validation_{metric[0]}'] for epoch in model_data]
                else:
                    # Calculate the average of the folds for each epoch
                    epoch_averages = {}
                    for entry in model_data:
                        fold = entry['fold']
                        epoch = entry['epoch']
                        if epoch not in epoch_averages:
                            epoch_averages[epoch] = {f'training_{metric[0]}': [],
                                                     f'validation_{metric[0]}': []
                                       }
                        epoch_averages[epoch][f'training_{metric[0]}'].append(entry[f'training_{metric[0]}'])
                        epoch_averages[epoch][f'validation_{metric[0]}'].append(entry[f'validation_{metric[0]}'])
                    for epoch, ep_metrics in epoch_averages.items():
                        avg_metrics = {}
                        for ep_metric, ep_values in ep_metrics.items():
                            avg_metrics['avg_' + ep_metric] = sum(ep_values) / len(ep_values)
                        epoch_averages[epoch] = avg_metrics
                    train_values = [epoch_averages[epoch][f'avg_training_{metric[0]}'] for epoch in epoch_averages]
                    val_values = [epoch_averages[epoch][f'avg_validation_{metric[0]}'] for epoch in epoch_averages]
                line_train, = ax.plot(range(1, len(train_values) + 1),
                                    train_values, marker='o', label=train_label)
                line_val, = ax.plot(range(1, len(val_values) + 1), val_values,
                                    marker='o', label=val_label, linestyle='dashed')

                lines.extend([line_train, line_val])
                labels.extend([train_label, val_label])

            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel(metric[1] + " (%)" if metric[0] != "loss" else metric[1], fontsize=14)
            if len(data) == 1:
                ax.set_title(f'{models_name[0]} {metric[1]} Train Results', fontsize=16)
            else:
                ax.set_title(f'{metric[1]} Train Results', fontsize=16)
            ax.legend(lines, labels, loc='best')

            # Add a description under the title
            #ax.text(0.5, -0.12, configuration, ha='center', va='center', transform=ax.transAxes, fontsize=11, color='black')

            # Save the plot to a file
            if not os.path.exists(os.path.join(script_directory, "results")):
                os.makedirs(os.path.join(script_directory, "results"))
            save_path = os.path.join(
                script_directory, "results", f"{save_plot_prefix}_train_{metric[0]}.png")
            plt.savefig(save_path)
            print(f"Plot saved as {save_path}")
        
        elif ("LOOCV" in validation_schemes and "K-FOLDCV" not in validation_schemes and "SPLIT" not in validation_schemes) or (
            "K-FOLDCV" in validation_schemes and "LOOCV" not in validation_schemes and "SPLIT" not in validation_schemes):
            num_folds = validation_settings[0]
            for fi in range(num_folds):
                _, ax = plt.subplots(figsize=(10, 7))
                lines = []
                labels = []
                for j, model_data in enumerate(data):
                    train_label = f"{models_name[j]} Train {metric[1]}"
                    val_label = f"{models_name[j]} Validation {metric[1]}"

                    fold_data = [entry for entry in model_data if entry["fold"] == fi+1]

                    train_values = [epoch[f'training_{metric[0]}'] for epoch in fold_data]
                    val_values = [epoch[f'validation_{metric[0]}'] for epoch in fold_data]

                    line_train, = ax.plot(range(1, len(train_values) + 1),
                                        train_values, marker='o', label=train_label)
                    line_val, = ax.plot(range(1, len(val_values) + 1), val_values,
                                        marker='o', label=val_label, linestyle='dashed')

                    lines.extend([line_train, line_val])
                    labels.extend([train_label, val_label])

                ax.set_xlabel('Epoch', fontsize=14)
                ax.set_ylabel(metric[1] + " (%)" if metric[0] != "loss" else metric[1], fontsize=14)
                if len(data) == 1:
                    ax.set_title(f'{models_name[0]} {metric[1]} Fold {fi+1} Train Results', fontsize=16)
                else:
                    ax.set_title(f'{metric[1]} Fold {fi+1} Train Results', fontsize=16)
                ax.legend(lines, labels, loc='best')

                # Add a description under the title
                #ax.text(0.5, -0.12, configuration, ha='center', va='center', transform=ax.transAxes, fontsize=11, color='black')

                # Save the plot to a file
                if not os.path.exists(os.path.join(script_directory, "results")):
                    os.makedirs(os.path.join(script_directory, "results"))
                save_path = os.path.join(
                    script_directory, "results", f"{save_plot_prefix}_train_{metric[0]}_fold_{fi+1}.png")
                plt.savefig(save_path)
                print(f"Plot saved as {save_path}")
        
        elif validation_schemes.count("SPLIT") == len(validation_schemes):
            _, ax = plt.subplots(figsize=(10, 7))
            lines = []
            labels = []
            for j, model_data in enumerate(data):
                train_label = f"{models_name[j]} Train {metric[1]}"
                val_label = f"{models_name[j]} Validation {metric[1]}"

                train_values = [epoch[f'training_{metric[0]}'] for epoch in model_data]
                val_values = [epoch[f'validation_{metric[0]}'] for epoch in model_data]

                line_train, = ax.plot(range(1, len(train_values) + 1),
                                    train_values, marker='o', label=train_label)
                line_val, = ax.plot(range(1, len(val_values) + 1), val_values,
                                    marker='o', label=val_label, linestyle='dashed')

                lines.extend([line_train, line_val])
                labels.extend([train_label, val_label])

            ax.set_xlabel('Epoch', fontsize=14)
            ax.set_ylabel(metric[1] + " (%)" if metric[0] != "loss" else metric[1], fontsize=14)
            if len(data) == 1:
                ax.set_title(f'{models_name[0]} {metric[1]} Train Results', fontsize=16)
            else:
                ax.set_title(f'{metric[1]} Train Results', fontsize=16)
            ax.legend(lines, labels, loc='best')

            # Add a description under the title
            #ax.text(0.5, -0.12, configuration, ha='center', va='center', transform=ax.transAxes, fontsize=11, color='black')

            # Save the plot to a file
            if not os.path.exists(os.path.join(script_directory, "results")):
                os.makedirs(os.path.join(script_directory, "results"))
            save_path = os.path.join(
                script_directory, "results", f"{save_plot_prefix}_train_{metric[0]}.png")
            plt.savefig(save_path)
            print(f"Plot saved as {save_path}")
        
def create_fold_line_plots(fold_data):
    pass # TODO

# ---CONFIGURATIONS--- #
# Folder names of the tests to compare
results_folders = ["ResNet18_2024-04-17_09-54-43"]
# Metrics to plot (name in the result file, plot label)
metrics = [('accuracy', 'Accuracy'), ('recall', 'Recall'), ('precision', 'Precision'), ('f1', 'F1'), ('auroc', 'AUROC'), ('loss', 'BCE Loss')]
# ---END OF CONFIGURATIONS--- #

models_name = [name.split("_")[0] for name in results_folders]
models_data = []
folds_data = []
validation_schemes = []
validation_settings = []
# Read configurations.json file
for results_folder in results_folders:
    with open(os.path.join(os.path.dirname(__file__), '..', 'results', results_folder, 'configurations.json'), 'r') as file:
        configurations = json.load(file)
    validation_scheme = configurations['validation_scheme']
    validation_setting = None
    if validation_scheme == 'LOOCV' or validation_scheme == 'K-FOLDCV':
        validation_setting = configurations['k_folds']
    else:
        validation_setting = configurations['split_ratio']
    validation_schemes.append(validation_scheme)
    validation_settings.append(validation_setting)

    train_val_data, fold_data = read_train_val_results(result_folder=results_folder,
                                  validation_scheme=validation_scheme
                                )
    models_data.append(train_val_data)
    folds_data.append(fold_data)

create_train_val_line_plots(metrics=metrics,
                             data=models_data,
                             models_name=models_name,
                             validation_schemes=validation_schemes,
                             validation_settings=validation_settings
                            )

for fold_data in folds_data:
    create_fold_line_plots(fold_data)