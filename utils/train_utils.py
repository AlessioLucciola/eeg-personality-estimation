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