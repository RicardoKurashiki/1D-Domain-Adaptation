import os
import random
import numpy
import torch

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_yaml(config):
    pass

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