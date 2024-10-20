from config import AMIGOS_FILES_DIR, AMIGOS_METADATA_FILE, ASCERTAIN_FILES_DIR, ASCERTAIN_METADATA_FILE, USE_DML, RESULTS_DIR
import numpy as np
import random
import torch
import json
import os

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"--RANDOM SEED-- Random seed set as {seed}")

def select_device():
    if USE_DML:
        import torch_directml
        device = torch_directml.device()
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('--DEVICE-- Using device: %s' % device)
    return device

def save_configurations(data_name, configurations):
    path = RESULTS_DIR + f"/{data_name}/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    results_file_path = path + 'configurations.json'
    with open(results_file_path, 'w') as json_file:
        json.dump(configurations, json_file, indent=2)

def get_configurations(model_path):
    conf_path = RESULTS_DIR + f"/{model_path}/configurations.json"
    configurations = None
    if os.path.exists(conf_path):
        print(
            "--Model-- Old configurations found. Using those configurations for training/testing the model.")
        with open(conf_path, 'r') as json_file:
            configurations = json.load(json_file)
    else:
        print("--Model-- Old configurations NOT found. Using configurations in the config for training/testing the model.")
    return configurations

def save_results(data_name, results, test=False):
    path = RESULTS_DIR + f"/{data_name}/results/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    results_file_path = path + 'test_results.json' if test else path + 'tr_val_results.json'
    if os.path.exists(results_file_path):
        final_results = None
        with open(results_file_path, 'r') as json_file:
            final_results = json.load(json_file)
        final_results.append(results)
        with open(results_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=2)
    else:
        final_results = [results]
        with open(results_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=2)

def save_fold_results(data_name, results):
    path = RESULTS_DIR + f"/{data_name}/results/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    results_file_path = path + 'fold_results.json'
    if os.path.exists(results_file_path):
        final_results = None
        with open(results_file_path, 'r') as json_file:
            final_results = json.load(json_file)
        final_results.append(results)
        with open(results_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=2)
    else:
        final_results = [results]
        with open(results_file_path, 'w') as json_file:
            json.dump(final_results, json_file, indent=2)


def save_model(data_name, model, fold=None, epoch=None, is_best=False):
    path = RESULTS_DIR + f"/{data_name}/models/"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    if is_best:
        torch.save(model.state_dict(), f'{path}/personality_estimation_best.pt')
    else:
        if fold != None:
            torch.save(model.state_dict(), f'{path}/personality_estimation_fold_{fold}_epoch_{epoch}.pt')
        else:
            torch.save(model.state_dict(), f'{path}/personality_estimation_{epoch}.pt')

def instantiate_dataset(dataset_name, apply_label_discretization, discretization_method):
    if dataset_name == "AMIGOS":
        from datasets.AMIGOS_dataset import AMIGOSDataset
        return AMIGOSDataset(
            apply_label_discretization=apply_label_discretization,
            discretization_method=discretization_method,
            data_path=AMIGOS_FILES_DIR,
            metadata_path=AMIGOS_METADATA_FILE
        )
    if dataset_name == "ASCERTAIN":
        from datasets.ASCERTAIN_dataset import ASCERTAINDataset
        return ASCERTAINDataset(
            apply_label_discretization=apply_label_discretization,
            discretization_method=discretization_method,
            data_path=ASCERTAIN_FILES_DIR,
            metadata_path=ASCERTAIN_METADATA_FILE
        )
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")
    
def resume_folds_metrics(dataset_name, fold=None, epoch=None):
    if epoch is None and fold is None:
        path = RESULTS_DIR + f"/{dataset_name}/results/fold_results.json"
        if os.path.exists(path):
            with open(path, 'r') as json_file:
                return json.load(json_file)
    else:
        path = RESULTS_DIR + f"/{dataset_name}/results/tr_val_results.json"
        if os.path.exists(path):
            with open(path, 'r') as json_file:
                all_results = json.load(json_file)
                filtered_results = []
                for result in all_results:
                    if result['fold'] == fold and result['epoch'] <= epoch:
                        filtered_results.append(result)
                return filtered_results
            
def save_starting_weights(weights, path):
    data_path = os.path.join(RESULTS_DIR, path)
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    torch.save(weights, data_path + f'/model_starting_weights.pt')

def resume_starting_weights(path):
    return torch.load(f'{RESULTS_DIR}/{path}/model_starting_weights.pt')