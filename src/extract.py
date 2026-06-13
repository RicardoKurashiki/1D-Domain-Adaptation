import os
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader

from .models import FeatureExtractor, Autoencoder

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

def extract_features(path: str, model: FeatureExtractor, data: DataLoader, data_label:str=None, verbose:bool=True):
    print(f"Loading model from {path}")
    model.load(path)
    model = model.to(device)

    model.eval()

    all_features = []
    all_labels = []
    pbar = tqdm(data, desc="EXTRACTION", leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            features = model(inputs)
        all_features.append(features.cpu().detach().numpy())
        all_labels.append(labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if data_label is not None:
        np.save(os.path.join(path, f"{data_label}_features.npy"), all_features)
        np.save(os.path.join(path, f"{data_label}_labels.npy"), all_labels)
    else:
        np.save(os.path.join(path, f"features.npy"), all_features)
        np.save(os.path.join(path, f"labels.npy"), all_labels)

    return all_features, all_labels

def align_features(path: str, model: Autoencoder, data: DataLoader, data_label: str = "target_aligned"):
    model.load(path)
    model = model.to(device)
    model.eval()

    all_features = []
    all_labels = []
    pbar = tqdm(data, desc="ALIGNMENT", leave=False)

    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.float().to(device)
            labels = labels.to(device)
            is_vae = model.arch == "variational_autoencoder"
            if is_vae:
                x_recon, z, mean, log_var = model(inputs)
            else:
                x_recon, z = model(inputs)
            all_features.append(x_recon.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    np.save(os.path.join(path, f"{data_label}_features.npy"), all_features)
    np.save(os.path.join(path, f"{data_label}_labels.npy"), all_labels)

    return all_features, all_labels