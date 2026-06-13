import os
import csv
import random
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

RESULTS_CSV = "src/experiments/results.csv"

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

def append_results(row: dict, csv_path: str = RESULTS_CSV):
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def get_dataloader(dataset, sampler=None, batch_size=32, shuffle=True):
    pin_memory = torch.cuda.is_available()
    if sampler is not None:
        # batch_sampler é mutuamente exclusivo com shuffle e batch_size
        return DataLoader(
            dataset,
            batch_sampler=sampler,
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

def plot_pca(path:str, features, labels, title:str="PCA", n_components=2, pca=None, prototypes=None, prototype_labels=None, axis_limits=None):
    if pca is None:
        pca = PCA(n_components=n_components)
        pca.fit(features)
    
    features = pca.transform(features)

    plt.figure(figsize=(10,8))
    scatter = plt.scatter(features[:,0], features[:,1], c=labels, alpha=0.3)
    plt.colorbar(scatter, label="Class")

    if prototypes is not None:
        centroids_2d = pca.transform(prototypes)
        for i, c in enumerate(centroids_2d):
            color = scatter.cmap(scatter.norm(prototype_labels[i])) if prototype_labels is not None else "red"
            label = f"Prototype class {prototype_labels[i]}" if prototype_labels is not None else f"Cluster {i}"
            plt.scatter(c[0], c[1], c=[color], marker="X", s=300, edgecolors="black", linewidths=3, label=label, zorder=5)
        plt.legend()

    if axis_limits is not None:
        xlim, ylim = axis_limits
        plt.xlim(xlim)
        plt.ylim(ylim)
    else:
        x_min, x_max = features[:, 0].min(), features[:, 0].max()
        y_min, y_max = features[:, 1].min(), features[:, 1].max()
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        if x_margin == 0: x_margin = 1.0
        if y_margin == 0: y_margin = 1.0
        axis_limits = ((x_min - x_margin, x_max + x_margin), (y_min - y_margin, y_max + y_margin))
        plt.xlim(axis_limits[0])
        plt.ylim(axis_limits[1])

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")

    output_path = os.path.join(path, f"{title.lower().replace(" ", "_")}_pca.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    
    return pca, axis_limits
    