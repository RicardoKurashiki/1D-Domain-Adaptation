import os
import csv
import json
import random
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from .constants import RESULTS_CSV, IMAGENET_MEAN, IMAGENET_STD

def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_transformations(config, split):
    backbone = config["model"].get("backbone")
    if backbone == "vit_lora":
        processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
        normalize = transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
        edge = processor.size.get("shortest_edge") or processor.size.get("height") or 224
        resize_train = [transforms.Resize((edge, edge))]
        resize_eval = [transforms.Resize((edge, edge))]
    else:
        normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        resize_train = [transforms.Resize(256), transforms.RandomCrop(224)]
        resize_eval = [transforms.Resize(256), transforms.CenterCrop(224)]

    aug_cfg = config.get("data_augmentation", {})
    apply_aug = split == "train" and aug_cfg.get("enabled", False)

    if apply_aug:
        rotation = aug_cfg.get("rotation_range", 0)
        shear    = aug_cfg.get("shear_range", 0.0)
        zoom     = aug_cfg.get("zoom_range", None)
        h_flip   = aug_cfg.get("horizontal_flip", False)
        v_flip   = aug_cfg.get("vertical_flip", False)

        t = list(resize_train)

        if h_flip:
            t.append(transforms.RandomHorizontalFlip())
        if v_flip:
            t.append(transforms.RandomVerticalFlip())

        if rotation or shear or zoom:
            t.append(transforms.RandomAffine(
                degrees=rotation or 0,
                shear=shear or None,
                scale=tuple(zoom) if zoom else None,
            ))

        t += [transforms.ToTensor(), normalize]
        return transforms.Compose(t)

    return transforms.Compose(resize_eval + [transforms.ToTensor(), normalize])

def append_results(row: dict, csv_path: str = RESULTS_CSV):
    key = "experiment"
    rows = []
    fieldnames = list(row.keys())

    if os.path.exists(csv_path):
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            rows = list(reader)

    for col in row.keys():
        if col not in fieldnames:
            fieldnames.append(col)

    replaced = False
    for i, existing in enumerate(rows):
        if str(existing.get(key, "")) == str(row.get(key, "")):
            rows[i] = row
            replaced = True
            break
    if not replaced:
        rows.append(row)

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({col: r.get(col, "") for col in fieldnames})


def save_results(row: dict, exp_dir: str, csv_path: str = RESULTS_CSV):
    append_results(row, csv_path)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "results.json"), "w") as f:
        json.dump(row, f, indent=2)

def get_dataloader(dataset, sampler=None, batch_size=32, shuffle=True):
    pin_memory = torch.cuda.is_available()
    if sampler is not None:
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
    