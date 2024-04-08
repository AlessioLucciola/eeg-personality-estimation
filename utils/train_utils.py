from config import LEARNING_RATE, REG
import time
import torch

def get_optimizer(optimizer_name, parameters, lr=LEARNING_RATE, weight_decay=REG):
    if optimizer_name == "Adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if optimizer_name == "AdamW":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Optimizer {optimizer_name} is not supported.")
    
def optimizer_zerp_grad(optimizer):
    optimizer.zero_grad(set_to_none=True)

def step(model, batch, isTraining, device):
    phase = "train" if isTraining else "val"
    eegs = batch["eeg_data"].to(device)
    labels = batch["labels"].to(device)
    ids = batch["ids"]
    starting_time = time.time()
    outputs = model(eegs, ids) # TO DO: Forward pass
    results = None # TO DO: Compute the results

def on_fit_end(logger):
    return logger.logs.groupby('epoch').min().sort_values(by='acc_mean_val', ascending=False).iloc[0:1, :]