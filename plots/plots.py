from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
from config import PLOTS_DIR
from math import ceil
import seaborn as sns
import pandas as pd
import math
import os

def plot_sample(eeg_data, rows_names, dataset_name, data_normalized=False, scale=25, title="EEG data sample"):
    cols = 4  # Number of columns for the subplots
    rows = len(rows_names) // cols + (len(rows_names) % cols != 0)  # Calculate number of rows, ensure coverage of all rows
    
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(scale * cols // 2, scale * rows // 4), tight_layout=True)
    full_title = f"{title} - {dataset_name} dataset"
    fig.suptitle(full_title, fontsize=24)
    
    for i_ax, ax in enumerate(axs.flat):
        if i_ax >= len(rows_names):
            ax.set_visible(False)
            continue
        
        ax.plot(eeg_data[:, i_ax])  # Plot the EEG data
        norm_string = "Before normalization" if not data_normalized else "After normalization"
        ax.set_title(f"{rows_names[i_ax]} channel - {norm_string}", fontsize=24)
        ax.set_xlabel('Sample (n)', fontsize=20)  # Set x-axis label
        ax.set_ylabel('Amplitude (µV)', fontsize=20)  # Set y-axis label
        ax.tick_params(axis='both', which='major', labelsize=16)  # Increase xtick and ytick label size
        ax.set_ylim(eeg_data[:, i_ax].min(), eeg_data[:, i_ax].max())
    
    # Determine the path to save the plot
    path_to_save_data = os.path.join(
        PLOTS_DIR, 
        "results", 
        dataset_name, 
        "eeg_sample_before_normalization" if not data_normalized else "eeg_sample_after_normalization"
    )
    if not os.path.exists(path_to_save_data):
        os.makedirs(path_to_save_data)
    
    save_path = os.path.join(path_to_save_data, f"{title}.png")
    plt.savefig(save_path)
    plt.close()  # Close the figure to release memory

def plot_amplitudes_distribution(eeg_data, rows_names, dataset_name, scale=10, title="Distribution of amplitudes"):
    cols = 7
    rows = ceil(len(rows_names) / cols)
    fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(scale * cols, scale * rows), tight_layout=True)
    full_title = f"{title} - {dataset_name} dataset"
    
    # Set the main title with increased size
    fig.suptitle(full_title, fontsize=30, y=1.02)
    
    for i_electrode, ax in enumerate(axs.flat):
        if i_electrode >= len(rows_names):
            ax.set_visible(False)
            continue
        
        # Plot histogram
        ax.hist(eeg_data[:, i_electrode], bins=32, color='b', alpha=0.7)
        
        # Set individual titles and labels
        ax.set_title(f"Amplitude distribution - Channel {rows_names[i_electrode]}", fontsize=28)
        ax.set_xlabel("Amplitude (µV)", fontsize=24)
        ax.set_ylabel("Count", fontsize=24)
        
        # Set y-axis to logarithmic scale
        ax.set_yscale("log")
        
        # Format x-axis labels
        ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
        
        # Increase the font size of the ticks
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
    
    # Normalize y-limits across all subplots
    max_ylim = max([ax.get_ylim()[-1] for ax in axs.flat if ax.get_visible()])
    for ax in axs.flat:
        if ax.get_visible():
            ax.set_ylim([None, max_ylim])
    
    # Save plot to file
    path_to_save_data = os.path.join(PLOTS_DIR, "results", dataset_name, "eeg_amplitude_distribution_after_normalization")
    if not os.path.exists(path_to_save_data):
        os.makedirs(path_to_save_data)
    save_path = os.path.join(path_to_save_data, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)  # Close the figure to avoid memory issues

def plot_subjects_distribution(subject_samples_num, dataset_name, title="Subjects distribution"):
    # Sort subject samples by number of samples in descending order
    sorted_subject_samples = sorted(subject_samples_num.items(), key=lambda x: x[1], reverse=True)
    subject_ids_samples = [x[0] for x in sorted_subject_samples]
    sample_counts = [x[1] for x in sorted_subject_samples]
    
    # Construct the full title
    full_title = f"{title} - {dataset_name} dataset"
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 5), tight_layout=True)
    
    # Plotting the barplot
    sns.barplot(x=subject_ids_samples, y=sample_counts, ax=ax)
    
    # Set title and labels with increased font size
    fig.suptitle(full_title, fontsize=24)
    ax.set_xlabel("Subject ID", fontsize=20)
    ax.set_ylabel("Number of Samples", fontsize=20)
    
    # Increase the font size of the ticks
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Define path to save the plot
    path_to_save_data = os.path.join(PLOTS_DIR, "results", dataset_name, "subjects_distribution")
    if not os.path.exists(path_to_save_data):
        os.makedirs(path_to_save_data)
    
    # Save the plot
    save_path = os.path.join(path_to_save_data, f"{title.replace(' ', '_').lower()}.png")
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to release memory

def plot_labels_distribution(labels, labels_num, discretized_labels, dataset_name, title="Distribution of labels", scale=10):
    title = f"{title} - {dataset_name} dataset"
    if discretized_labels:
        fig, ax = plt.subplots(figsize=(scale, scale))
        ax.pie(labels_num.values(), labels=labels, autopct='%1.1f%%', shadow=False)
        fig.suptitle(title)
        ax.axis('equal')
        path_to_save_data = os.path.join(PLOTS_DIR, "results", dataset_name, "labels_distribution")
        if not os.path.exists(path_to_save_data):
            os.makedirs(path_to_save_data)
        save_path = os.path.join(path_to_save_data, f"{title}.png")
        plt.savefig(save_path)
    else:
        # TO DO: Eventually implement the plot for continuous labels
        return
    
def plot_mel_spectrogram(eeg_data, spectrogram_function, rows_name, dataset_name, title="Mel spectrogram of EEG data sample", scale=3):
    spectrogram = spectrogram_function(eeg_data)
    num_spectrograms = spectrogram.shape[0]
    lines = int(math.ceil(math.sqrt(num_spectrograms)))
    
    # Increased figsize for larger subplots
    fig, axs = plt.subplots(nrows=lines, ncols=lines, figsize=(lines * scale * 2, lines * scale * 1.5), tight_layout=True)
    min_value, max_value = spectrogram.min(), spectrogram.max()
    
    # Increase the size of the figure title
    fig.suptitle(f"{title} - {dataset_name} dataset", fontsize=24)

    for i_ax, ax in enumerate(axs.flat):
        if i_ax < num_spectrograms:
            im = ax.imshow(spectrogram[i_ax, :, :], vmin=min_value, vmax=max_value, aspect="auto", cmap=plt.get_cmap("hot"))
            
            # Create a colorbar for each subplot
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
            
            # Set title and labels with increased font size
            ax.set_title(f"Mel Spectrogram - Channel {rows_name[i_ax]}", fontsize=20)
            ax.set_xlabel("Time (s)", fontsize=16)
            ax.set_ylabel("Mels (n)", fontsize=16)
            
            # Increase tick label size
            ax.tick_params(axis='both', which='major', labelsize=16)
            
            ax.invert_yaxis()  # Invert y-axis for Mel spectrogram
        else:
            ax.axis('off')  # Turn off the axes for empty subplots
    
    # Define path to save the plot
    path_to_save_data = os.path.join(PLOTS_DIR, "results", dataset_name, "mel_spectrograms")
    if not os.path.exists(path_to_save_data):
        os.makedirs(path_to_save_data)
    save_path = os.path.join(path_to_save_data, f"{title}.png")
    
    plt.savefig(save_path)
    plt.close(fig)  # Close the figure to avoid memory issues

def plot_trait_distribution(metadata_df, trait_name, mean_value, dataset_name, discretization_type, title="Trait distribution"):
    trait_values = metadata_df[trait_name]
    
    # Group values into chunks
    bins = [1, 2, 3, 4, 5, 6, 7]  # Define the bins for grouping
    labels = ['1', '2', '3', '4', '5', '6']  # Labels for the bins
    
    # Use pd.cut with include_lowest=True to include the lowest bin edge
    grouped_counts = trait_values.groupby(pd.cut(trait_values, bins=bins, labels=labels, include_lowest=True), observed=True).count()
    
    # Fill in missing bins with 0 counts
    grouped_counts = grouped_counts.reindex(labels, fill_value=0)
    
    plt.figure(figsize=(10, 6))  # Adjust figure size as needed
    
    # Plot bar chart
    ax = grouped_counts.plot(kind='bar', color='b', width=0.8, alpha=0.7)
    
    # Plot mean line
    ax.axvline(x=mean_value-1, linestyle='--', color='r', label='Mean')

    plt.annotate(discretization_type, xy=(0.5, -0.25), xycoords='axes fraction', ha='center', va='center',
                 fontsize=16, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white", alpha=0.7))
    
    # Set title and labels with increased font size
    plt.title(f'Distribution of Trait: {trait_name}', fontsize=24)
    plt.xlabel('Trait Value Groups', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    
    # Set x-axis tick labels with increased font size
    plt.xticks(range(len(labels)), labels, rotation=0, fontsize=16)
    plt.yticks(fontsize=16)
    
    # Set legend font size
    plt.legend(fontsize=16)
    
    # Grid lines
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot to file
    path_to_save_data = os.path.join(PLOTS_DIR, "results", dataset_name, "trait_distribution")
    if not os.path.exists(path_to_save_data):
        os.makedirs(path_to_save_data)
    save_path = os.path.join(path_to_save_data, f"{title}_{trait_name}_{discretization_type}.png")
    plt.savefig(save_path)
    plt.close()  # Close the figure to release memory