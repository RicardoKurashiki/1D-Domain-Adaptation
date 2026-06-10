from .datasets import FeatureSpaceDataset

def k_means():
    pass

def k_medoids():
    pass

def k_center():
    pass

def enn(data, labels):
    from imblearn.under_sampling import EditedNearestNeighbours
    sampler = EditedNearestNeighbours()
    return sampler.fit_resample(data, labels)

def renn(data, labels):
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours
    sampler = RepeatedEditedNearestNeighbours()
    return sampler.fit_resample(data, labels)

def tomek_links(data, labels):
    from imblearn.under_sampling import TomekLinks
    sampler = TomekLinks()
    return sampler.fit_resample(data, labels)

def run_undersampling(data, labels, method:str):
    import torch
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    match method:
        case "enn":
            return enn(data, labels)
        case "renn":
            return renn(data, labels)
        case "tomek_links":
            return tomek_links(data, labels)
        case _:
            return data, labels

def run_clustering(data, method:str):
    match method:
        case "k_means":
            return k_means()
        case "k_medoids":
            return k_medoids()
        case "k_center":
            return k_center()
        case _:
            print(f"Nenhum método \"{method}\" de clustering encontrado")

def run(dataset, undersampling:str=None, clustering:str="k_means"):
    print("Normal: ", len(dataset))
    features, labels = run_undersampling(dataset.features, dataset.labels, method=undersampling)
    dataset_und = FeatureSpaceDataset(features, labels)
    print("Undersampled: ", len(dataset_und))
    # prototypes = run_clustering(data, method=clustering)