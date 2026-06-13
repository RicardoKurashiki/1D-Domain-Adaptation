import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import numpy as np
import torch.nn as nn
import torch.optim as optim

from torchinfo import summary

from src.datasets import KermanyDataset, RSNADataset, FeatureSpaceDataset, MiniBatchSampler
from src.models import FeatureExtractor, ClassificationHead, ClassifierModel, Autoencoder
from src.configuration import Configuration, EarlyStoppingConfig, ReduceLROnPlateauConfig, AutoencoderConfiguration
from src.extract import extract_features, align_features
from src.losses import CenterLoss
from src import utils, stage1, stage2, prototype

SEED = 42
BATCH_SIZE = 32
EPOCHS = 500
N_CLUSTERS = 1
LR = 0.0001

SOURCE_DATASET = RSNADataset
TARGET_DATASET = KermanyDataset
PATH = "./results/RSNA"

if __name__ == '__main__':
    utils.set_seed(SEED)

    train_dataset = SOURCE_DATASET(split="train")
    val_dataset = SOURCE_DATASET(split="val")
    test_dataset = TARGET_DATASET(split="test")

    print("Train: ", len(train_dataset))
    print("Val: ", len(val_dataset))
    print("Test: ", len(test_dataset))

    sampler = MiniBatchSampler(train_dataset, batch_size=BATCH_SIZE)

    extractor = FeatureExtractor(backbone="resnet18", unfrozen_layers=None)
    classifier = ClassificationHead(in_features=extractor.num_ftrs, out_features=train_dataset.n_classes)
    model = ClassifierModel(extractor, classifier)

    summary(model, depth=10, col_names=["trainable"])

    print(f"\nExtractor size: {extractor.get_trainable_params()} params | {extractor.get_model_size():.3f} MB")
    print(f"Classifier size: {classifier.get_trainable_params()} params | {classifier.get_model_size():.3f} MB")
    print(f"Model size: {model.get_trainable_params()} params | {model.get_model_size():.3f} MB")

    train_data = utils.get_dataloader(train_dataset, sampler=sampler, shuffle=False)
    val_data = utils.get_dataloader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_data = utils.get_dataloader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    es = EarlyStoppingConfig()
    reduce_lr = ReduceLROnPlateauConfig(optimizer=optimizer, patience=5)

    training_config = Configuration(epochs=EPOCHS, optimizer=optimizer, criterion=criterion, early_stopping=es, reduce_lr=reduce_lr)

    stage1.train(path=PATH, model=model, train_data=train_data, val_data=val_data, config=training_config)
    stage1.test(path=PATH, model=model, data=test_data, criterion=criterion)
    
    src_features, src_labels = extract_features(path=PATH, model=extractor, data=train_data, data_label="source")
    
    src_features = np.load(os.path.join(PATH, f"source_features.npy"))
    src_labels = np.load(os.path.join(PATH, f"source_labels.npy"))
    
    feature_dataset = FeatureSpaceDataset(features=src_features, labels=src_labels)
    feature_data = utils.get_dataloader(feature_dataset, batch_size=BATCH_SIZE, shuffle=False)
    pt_dataset = prototype.run(dataset=feature_dataset, undersampling="enn", clustering="k_means", k=N_CLUSTERS, seed=SEED)

    autoencoder = Autoencoder(arch="variational_autoencoder", input_dim=feature_dataset.features_dim, hidden_dim=feature_dataset.features_dim//2, latent_dim=feature_dataset.features_dim//4)

    center_loss = CenterLoss(num_classes=train_dataset.n_classes, feat_dim=feature_dataset.features_dim, initial_centers=pt_dataset.ordered_centroids())
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=LR)
    ae_config = AutoencoderConfiguration(epochs=EPOCHS, optimizer=ae_optimizer, early_stopping=es, reduce_lr=None, alignment_loss=center_loss, align_weight=0.9, kl_weight=0.1, reconstruction_weight=0.0)
    

    tgt_train_dataset = TARGET_DATASET(split="train")
    tgt_train_data = utils.get_dataloader(tgt_train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    tgt_val_dataset = TARGET_DATASET(split="val")
    tgt_val_data = utils.get_dataloader(tgt_val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    tgt_test_dataset = TARGET_DATASET(split="test")
    tgt_test_data = utils.get_dataloader(tgt_test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # tgt_features, tgt_labels = extract_features(path=PATH, model=extractor, data=tgt_train_data, data_label="target")
    tgt_features = np.load(os.path.join(PATH, f"target_features.npy"))
    tgt_labels = np.load(os.path.join(PATH, f"target_labels.npy"))
    
    tgt_feature_dataset = FeatureSpaceDataset(features=tgt_features, labels=tgt_labels)
    tgt_feature_data = utils.get_dataloader(tgt_feature_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # tgt_val_features, tgt_val_labels = extract_features(path=PATH, model=extractor, data=tgt_val_data, data_label="target_val")
    tgt_val_features = np.load(os.path.join(PATH, f"target_val_features.npy"))
    tgt_val_labels = np.load(os.path.join(PATH, f"target_val_labels.npy"))
    
    tgt_val_feature_dataset = FeatureSpaceDataset(features=tgt_val_features, labels=tgt_val_labels)
    tgt_val_feature_data = utils.get_dataloader(tgt_val_feature_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # tgt_test_features, tgt_test_labels = extract_features(path=PATH, model=extractor, data=tgt_test_data, data_label="target_test")
    tgt_test_features = np.load(os.path.join(PATH, f"target_test_features.npy"))
    tgt_test_labels = np.load(os.path.join(PATH, f"target_test_labels.npy"))

    tgt_test_feature_dataset = FeatureSpaceDataset(features=tgt_test_features, labels=tgt_test_labels)
    tgt_test_feature_data = utils.get_dataloader(tgt_test_feature_dataset, batch_size=BATCH_SIZE, shuffle=False)


    source_pca, axis_limits = utils.plot_pca(
        path=PATH, features=src_features, labels=src_labels,
        title="Source Domain",
        prototypes=pt_dataset.features, prototype_labels=pt_dataset.labels
    )

    utils.plot_pca(
        path=PATH, features=tgt_test_features, labels=tgt_test_labels,
        title="Target Domain (Source PCA)",
        pca=source_pca,
        prototypes=pt_dataset.features, prototype_labels=pt_dataset.labels,
        axis_limits=axis_limits
    )

    # stage2.train(path=PATH, model=autoencoder, train_data=tgt_feature_data, val_data=tgt_val_feature_data, config=ae_config)
    aligned_features, aligned_labels = align_features(path=PATH, model=autoencoder, data=tgt_test_feature_data, data_label="target_aligned")
    aligned_dataset = FeatureSpaceDataset(features=aligned_features, labels=aligned_labels)
    aligned_data = utils.get_dataloader(aligned_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- PCA 3: Aligned target no PCA do source, com protótipos ---
    utils.plot_pca(
        path=PATH, features=aligned_features, labels=aligned_labels,
        title="Aligned Target (Source PCA)",
        pca=source_pca,
        prototypes=pt_dataset.features, prototype_labels=pt_dataset.labels,
        axis_limits=axis_limits
    )

    # --- PCA 4: Source + Target (pré-alinhamento) lado a lado no mesmo espaço ---
    combined_features = np.concatenate([src_features, tgt_test_features], axis=0)
    combined_labels = np.concatenate([np.zeros(len(src_features)), np.ones(len(tgt_test_features))], axis=0)
    utils.plot_pca(
        path=PATH, features=combined_features, labels=combined_labels,
        title="Source vs Target (domain shift)",
        pca=source_pca,
        axis_limits=axis_limits
    )

    # --- PCA 5: Source + Aligned Target ---
    combined_aligned = np.concatenate([src_features, aligned_features], axis=0)
    utils.plot_pca(
        path=PATH, features=combined_aligned, labels=combined_labels,
        title="Source vs Aligned Target",
        pca=source_pca,
        axis_limits=axis_limits
    )

    stage2.test(path=PATH, model=classifier, data=tgt_test_feature_data, criterion=criterion)
    stage2.test(path=PATH, model=classifier, data=aligned_data, criterion=criterion)