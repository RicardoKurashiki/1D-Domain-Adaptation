import os
import random
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_data_augmentation(config, split):
    pass

def create_experiment_dir(config_path):
    exp_dir = os.path.dirname(config_path)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    prototypes_dir = os.path.join(exp_dir, "prototypes")
    if not os.path.exists(prototypes_dir):
        os.makedirs(prototypes_dir)
    latents_dir = os.path.join(exp_dir, "latents")
    if not os.path.exists(latents_dir):
        os.makedirs(latents_dir)

def update_registry(registry_path, row):
    # se o arquivo não existe, cria com header
    # se já existe, verifica se o exp_id já tem linha (update) ou adiciona
    pass

def get_dataloader(dataset, sampler=None, batch_size=32, shuffle=True):
    pin_memory = torch.cuda.is_available()
    if sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=2
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=2
    )