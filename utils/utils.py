from config import AMIGOS_FILES_DIR, AMIGOS_METADATA_FILE, USE_DML
import random
import numpy as np
import torch
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

def instantiate_dataset(dataset_name):
    if dataset_name == "AMIGOS":
        from datasets.AMIGOS_dataset import AMIGOSDataset
        return AMIGOSDataset(data_path=AMIGOS_FILES_DIR, metadata_path=AMIGOS_METADATA_FILE)
    else:
        raise ValueError(f"Dataset {dataset_name} is not supported.")