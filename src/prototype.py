import numpy as np

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

from .datasets import FeatureSpaceDataset
from .clustering import KCenterGreedy

def k_means(features, labels, n_clusters, seed):
    clusterer = KMeans(n_clusters=n_clusters, random_state=seed)
    clusterer.fit(features)
    centroids = clusterer.cluster_centers_
    labels = np.array([
            np.bincount(labels[clusterer.labels_ == c].astype(int)).argmax()
            for c in range(n_clusters)
        ])
    return centroids, labels

def k_medoids(features, labels, n_clusters, seed):
    clusterer = KMedoids(n_clusters=n_clusters, random_state=seed, init="k-medoids++")
    clusterer.fit(features)
    centroids = clusterer.cluster_centers_
    labels = np.array([
            np.bincount(labels[clusterer.labels_ == c].astype(int)).argmax()
            for c in range(n_clusters)
        ])
    return centroids, labels

def k_center(features, labels, n_clusters, seed):
    clusterer = KCenterGreedy(n_clusters=n_clusters, random_state=seed)
    clusterer.fit(features)
    centroids = clusterer.cluster_centers_
    labels = np.array([
            np.bincount(labels[clusterer.labels_ == c].astype(int)).argmax()
            for c in range(n_clusters)
        ])
    return centroids, labels

def enn(data, labels):
    from imblearn.under_sampling import EditedNearestNeighbours
    sampler = EditedNearestNeighbours(n_neighbors=5)
    return sampler.fit_resample(data, labels)

def renn(data, labels):
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    sampler = RepeatedEditedNearestNeighbours(n_neighbors=5, max_iter=500)
    return sampler.fit_resample(data, labels)

def tomek_links(data, labels):
    from imblearn.under_sampling import TomekLinks
    sampler = TomekLinks()
    return sampler.fit_resample(data, labels)

def run_undersampling(data, labels, method:str):
    match method:
        case "enn":
            return enn(data, labels)
        case "renn":
            return renn(data, labels)
        case "tomek_links":
            return tomek_links(data, labels)
        case _:
            return data, labels

def run_clustering(data, labels, method:str, n_clusters:int, seed:int):
    match method:
        case "k_means":
            return k_means(data, labels, n_clusters, seed)
        case "k_medoids":
            return k_medoids(data, labels, n_clusters, seed)
        case "k_center":
            return k_center(data, labels, n_clusters, seed)
        case _:
            print(f"Nenhum método \"{method}\" de clustering encontrado")

def run(dataset, undersampling:str=None, clustering:str="k_means", k:int=2, seed:int=42):
    features, labels = run_undersampling(dataset.features, dataset.labels, method=undersampling)
    prototypes_features, prototypes_labels = run_clustering(features, labels, method=clustering, n_clusters=k, seed=seed)

    return FeatureSpaceDataset(features=prototypes_features, labels=prototypes_labels)