from shared.constants import amigos_labels_reverse
import matplotlib.pyplot as plt
from config import PLOTS_DIR
from math import ceil
import seaborn as sns
import os

def plot_sample(eeg_data, rows_names, dataset_name, data_normalized=False, scale=25, title="EEG data sample"):
    cols = 4
    rows = len(rows_names) // cols
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(scale * cols // 2, scale * rows // 4), tight_layout=True)
    title = f"{title} - {dataset_name} dataset"
    fig.suptitle(title)
    for i_ax, ax in enumerate(axs.flat):
        if i_ax >= len(rows_names):
            ax.set_visible(False)
            continue
        axs.flat[i_ax].plot(eeg_data[:, i_ax])
        axs.flat[i_ax].set_title(rows_names[i_ax])
        axs.flat[i_ax].set_ylim(eeg_data[:, i_ax].min(), eeg_data[:, i_ax].max())
    
    path_to_save_data = os.path.join(PLOTS_DIR, "results", dataset_name, "eeg_sample_before_normalization") if not data_normalized else os.path.join(PLOTS_DIR, "results", dataset_name, "eeg_sample_after_normalization")
    if not os.path.exists(path_to_save_data):
        os.makedirs(path_to_save_data)
    save_path = os.path.join(path_to_save_data, f"{title}.png")
    plt.savefig(save_path)

def plot_amplitudes_distribution(eeg_data, rows_names, dataset_name, scale=10, title="Distribution of amplitudes"):
    cols: int = 7
    rows: int = ceil(len(rows_names) / cols)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(scale * cols, scale * rows), tight_layout=True)
    title = f"{title} - {dataset_name} dataset"
    fig.suptitle(title)
    for i_electrode, ax in enumerate(axs.flat):
        if i_electrode >= len(rows_names):
            ax.set_visible(False)
            continue
        ax.hist(eeg_data[:, i_electrode], bins=32)
        ax.set_title(rows_names[i_electrode])
        ax.set_xlabel("mV")
        ax.set_ylabel("count")
        ax.set_yscale("log")
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    max_ylim = max([ax.get_ylim()[-1] for ax in axs.flat])
    for ax in axs.flat:
        ax.set_ylim([None, max_ylim])
    path_to_save_data = os.path.join(PLOTS_DIR, "results", dataset_name, "eeg_amplitude_distribution_after_normalization")
    if not os.path.exists(path_to_save_data):
        os.makedirs(path_to_save_data)
    save_path = os.path.join(path_to_save_data, f"{title}.png")
    plt.savefig(save_path)

def plot_subjects_distribution(subject_samples_num, dataset_name, title="Subjects distribution"):
    sorted_subject_samples = sorted(subject_samples_num.items(), key=lambda x: x[1], reverse=True)
    subject_ids_samples = [x[0] for x in sorted_subject_samples]
    sample_counts = [x[1] for x in sorted_subject_samples]
    title = f"{title} - {dataset_name} dataset"
    _, ax = plt.subplots(1, 1, figsize=(15, 5), tight_layout=True)
    sns.barplot(x=subject_ids_samples, y=sample_counts, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Subject ID")
    ax.set_ylabel("Number of Samples")
    path_to_save_data = os.path.join(PLOTS_DIR, "results", dataset_name, "subjects_distribution")
    if not os.path.exists(path_to_save_data):
        os.makedirs(path_to_save_data)
    save_path = os.path.join(path_to_save_data, f"{title}.png")
    plt.savefig(save_path)

def plot_labels_distribution(labels, labels_num, discretized_labels, dataset_name, title="Distribution of labels", scale=6):
    title = f"{title} - {dataset_name} dataset"
    if discretized_labels:
        _, ax = plt.subplots(figsize=(scale, scale))
        ax.pie(labels_num.values(), labels=labels, autopct='%1.1f%%', shadow=False)
        ax.set_title(title)
        ax.axis('equal')
        path_to_save_data = os.path.join(PLOTS_DIR, "results", dataset_name, "labels_distribution")
        if not os.path.exists(path_to_save_data):
            os.makedirs(path_to_save_data)
        save_path = os.path.join(path_to_save_data, f"{title}.png")
        plt.savefig(save_path)
    else:
        # TO DO: Eventually implement the plot for continuous labels
        return