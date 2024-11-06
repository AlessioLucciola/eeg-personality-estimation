# TA-EPE: A Transformer-based Approach to EEG Personality Estimation
Personality estimation involves predicting individual personality traits using various data sources, such as physiological signals, behavioral patterns, or brain activity. This task is intriguing because it can provide deeper insights into human behavior, enhance personalized user experiences, and improve mental health interventions.
This project proposes an innovative approach to personality estimation from EEG data using Transformer architectures. The main contribution is demonstrating the effectiveness of deep learning, particularly Transformers, in this complex task and exploring various validation schemes to ensure robust performance comparison. While EEG-based personality prediction is challenging due to the intricacies of brain signals, this work highlights the potential of advanced neural networks to improve accuracy.
The results demonstrate that our proposed approach significantly outperforms existing methods. To ensure a fair comparison with the related works, two validation schemes have been employed: Leave One Out (LOOCV) and K-FOLD cross-validation. With the LOO validation scheme on the AMIGOS dataset, our EEGTransformer achieved an impressive F1 score of 81.72%, a substantial improvement over the 43.40% F1 score of the current state-of-the-art, marking a difference of 38.32 percentage points. Under the KFOLDCV scheme, our model attained an accuracy of 99.30%, surpassing the best-reported accuracy of 90.58% by 8.72 percentage points. For the ASCERTAIN dataset, our EEGTransformer achieved an F1 score of 89.39% with LOO validation, significantly exceeding the 43.40% F1 score of the previous leading method, representing a remarkable improvement of 45.99 percentage points. This research not only highlights the effectiveness of our approach but also paves the way for future advancements in EEG-based personality prediction.

You can find more details in the [project report](https://github.com/AlessioLucciola/eeg-personality-estimation/blob/main/report.pdf).

## Installation
This project uses Python 3.10.6. To create the environment using conda do:

```
conda env create -f environment.yaml
conda activate personality-estimation
```
## Project structure
The repository is organized as follows:
- dataloaders: This directory contains custom PyTorch data loaders designed for splitting the data based on the chosen validation scheme and applying data augmentation to the training set;
- datasets: Here, custom PyTorch datasets, AMIGOS_dataset and ASCERTAIN_dataset, are implemented to load and standardize data across their respective datasets. These inherit from the EEG_classification_dataset module, which is responsible for data preprocessing;
- models and train_modules: The models directory defines the structure of various models, while train_modules includes the components necessary to initiate the training process. To train a model, execute the relevant script in train_modules (e.g., python -m train_modules.eeg_transformer );
- utils and shared: These directories contain utility functions and constants used throughout the application;
- data: This directory stores the datasets along with their associated metadata, including labels;
- plots: Contains functions for generating plots;
- results: This folder is designated for storing both checkpoints and metrics generated by each model. When a model training session starts, a new subfolder is created under results corresponding to the test. Several .json files are generated within this subfolder: configuration.json (storing the training hyperparameters), train_val_results.json (containing metric values for each epoch), and fold_results.json (providing metrics for each fold in k-fold and LOO validation schemes). Model checkpoints for each epoch are saved under results/test_name/models;
- config.py: This file allows users to define all training hyperparameters, enabling comprehensive control over experiments without modifying the core code. This setup facilitates rapid testing of various configurations.

## Data
The datasets used in this project are [AMIGOS](https://www.eecs.qmul.ac.uk/mmv/datasets/amigos/) and [ASCERTAIN](https://ascertain-dataset.github.io/). You can download the datasets from the official websites.

Inside the `data` folder, there should be these elements:
- amigos
  - Participant_Personality.xlsx
  - files
    - Data_Preprocessed_P01.mat
    - Data_Preprocessed_P02.mat
    - ...
    - Data_Preprocessed_P40.mat
- ascertain
  - Personality_Details.xls
  - Data_Evaluation.xls
  - files
      - Movie_P01
        - EEG_Clip1.mat
        - EEG_Clip2.mat
        - ...
        - EEG_Clip36.mat
      - Movie_P02
        - EEG_Clip1.mat
        - EEG_Clip2.mat
        - ...
        - EEG_Clip36.mat
      - ...
      - Movie_P58
        - EEG_Clip1.mat
        - EEG_Clip2.mat
        - ...
        - EEG_Clip36.mat

Where xls (or xlsx) files contain the metadata and mat files contain the EEG signals. It is not mandatory to download both datasets; you can also use only one, but the structure of each dataset must be as described above.

# Training
Set the hyperparameters in the `config.py` file and run the training process with `python -m train_modules.model_name` where model_name can be `eeg_transformer`, `resnet18` or `custom_cnn` (e.g. `python -m train_modules.eeg_transformer`). Results will be saved in the `results` folder if `SAVE_RESULTS` is True. Checkpoints will be saved in the `results` folder as well if `SAVE_MODELS` is True. Plots will be saved in the `plots` folder if `MAKE_PLOTS` is True.


