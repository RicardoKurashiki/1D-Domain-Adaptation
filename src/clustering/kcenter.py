import numpy as np
from scipy.spatial.distance import cdist

class KCenterGreedy:
    def __init__(self, n_clusters, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X)
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        first = rng.integers(n_samples)
        centers_idx = [first]
        min_dist = cdist(X, X[first:first + 1], metric="euclidean").ravel()

        for _ in range(1, self.n_clusters):
            new_idx = int(min_dist.argmax())
            centers_idx.append(new_idx)
            new_dist = cdist(X, X[new_idx:new_idx + 1], metric="euclidean").ravel()
            min_dist = np.minimum(min_dist, new_dist)

        centers = X[centers_idx].copy()
        self.cluster_centers_ = centers
        self.labels_ = cdist(X, centers, metric="euclidean").argmin(axis=1)
        return self