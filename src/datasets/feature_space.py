import torch
import numpy as np
from torch.utils.data import Dataset

class FeatureSpaceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.features_dim = features.shape[1]

    def ordered_centroids(self):
        unique_labels = np.unique(self.labels)
        centroids = []
        for lbl in sorted(unique_labels):
            mask = self.labels == lbl
            if mask.sum() > 1:
                centroid = self.features[mask].mean(axis=0)
            else:
                centroid = self.features[mask][0]
            centroids.append(centroid)
        return torch.tensor(np.stack(centroids), dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])