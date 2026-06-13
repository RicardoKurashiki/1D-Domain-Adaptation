import os
from datetime import datetime

import yaml
import numpy as np
import torch.nn as nn
import torch.optim as optim

from .datasets import KermanyDataset, RSNADataset, FeatureSpaceDataset, MiniBatchSampler
from .models import FeatureExtractor, ClassificationHead, ClassifierModel, Autoencoder
from .configuration import Configuration, EarlyStoppingConfig, ReduceLROnPlateauConfig, AutoencoderConfiguration
from .extract import extract_features, align_features
from .losses import CenterLoss
from . import utils, stage1, stage2, prototype

DATASET_MAP = {
    "rsna": RSNADataset,
    "kermany": KermanyDataset,
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_stage1(config: dict, exp_dir: str, force: bool):
    utils.set_seed(config["seed"])

    SourceDataset = DATASET_MAP[config["source_dataset"]]
    batch_size = config["batch_size"]

    train_dataset = SourceDataset(split="train")
    val_dataset = SourceDataset(split="val")
    test_dataset = SourceDataset(split="test")

    sampler = MiniBatchSampler(train_dataset, batch_size=batch_size)
    train_data = utils.get_dataloader(train_dataset, sampler=sampler, shuffle=False)
    val_data = utils.get_dataloader(val_dataset, batch_size=batch_size, shuffle=False)
    test_data = utils.get_dataloader(test_dataset, batch_size=batch_size, shuffle=False)

    backbone = config["model"]["backbone"]
    unfrozen_layers = config["model"].get("unfrozen_layers")
    extractor = FeatureExtractor(backbone=backbone, unfrozen_layers=unfrozen_layers)
    classifier = ClassificationHead(in_features=extractor.num_ftrs, out_features=train_dataset.n_classes)
    model = ClassifierModel(extractor, classifier)

    criterion = nn.CrossEntropyLoss()
    lr = config["lr"]
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    es_cfg = config.get("early_stopping", {})
    es = EarlyStoppingConfig(patience=es_cfg.get("patience", 10), start_epoch=es_cfg.get("start_epoch", 0))

    rl_cfg = config.get("reduce_lr", {})
    reduce_lr = ReduceLROnPlateauConfig(optimizer=optimizer, patience=rl_cfg.get("patience", 10), factor=rl_cfg.get("factor", 0.1))

    training_config = Configuration(
        epochs=config["epochs"],
        optimizer=optimizer,
        criterion=criterion,
        early_stopping=es,
        reduce_lr=reduce_lr,
    )

    weights_exist = os.path.exists(os.path.join(exp_dir, "extractor_weight.pt"))
    if not weights_exist or force:
        stage1.train(path=exp_dir, model=model, train_data=train_data, val_data=val_data, config=training_config)
    else:
        print(f"Weights found in {exp_dir}, skipping training.")
        model.load(exp_dir)

    test_loss, test_acc, test_precision, test_recall, test_f1 = stage1.test(
        path=exp_dir, model=model, data=test_data, criterion=criterion
    )

    row = {
        "experiment": exp_dir,
        "stage": 1,
        "timestamp": datetime.now().isoformat(),
        "seed": config["seed"],
        "backbone": backbone,
        "unfrozen_layers": str(unfrozen_layers),
        "source_dataset": config["source_dataset"],
        "target_dataset": "",
        "batch_size": batch_size,
        "lr": lr,
        "epochs": config["epochs"],
        "es_patience": es_cfg.get("patience", 10),
        "reduce_lr_patience": rl_cfg.get("patience", 10),
        "n_clusters": "",
        "clustering": "",
        "undersampling": "",
        "ae_arch": "",
        "align_weight": "",
        "kl_weight": "",
        "reconstruction_weight": "",
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "test_precision": round(test_precision, 6),
        "test_recall": round(test_recall, 6),
        "test_f1": round(test_f1, 6),
        "aligned_test_loss": "",
        "aligned_test_acc": "",
        "aligned_test_precision": "",
        "aligned_test_recall": "",
        "aligned_test_f1": "",
    }
    utils.append_results(row)
    print(f"\nResults saved to {utils.RESULTS_CSV}")


def run_stage2(config: dict, exp_dir: str, force: bool):
    utils.set_seed(config["seed"])

    SourceDataset = DATASET_MAP[config["source_dataset"]]
    TargetDataset = DATASET_MAP[config["target_dataset"]]
    batch_size = config["batch_size"]

    stage1_config_path = config["stage1_config"]
    stage1_cfg = load_config(stage1_config_path)
    stage1_exp_dir = os.path.dirname(os.path.abspath(stage1_config_path))

    backbone = stage1_cfg["model"]["backbone"]
    unfrozen_layers = stage1_cfg["model"].get("unfrozen_layers")
    extractor = FeatureExtractor(backbone=backbone, unfrozen_layers=unfrozen_layers)
    classifier = ClassificationHead(in_features=extractor.num_ftrs, out_features=2)
    model = ClassifierModel(extractor, classifier)
    model.load(stage1_exp_dir)

    src_train_dataset = SourceDataset(split="train")
    src_train_data = utils.get_dataloader(src_train_dataset, batch_size=batch_size, shuffle=False)

    src_features_path = os.path.join(exp_dir, "source_features.npy")
    if not os.path.exists(src_features_path) or force:
        src_features, src_labels = extract_features(path=exp_dir, model=extractor, data=src_train_data, data_label="source")
    else:
        src_features = np.load(src_features_path)
        src_labels = np.load(os.path.join(exp_dir, "source_labels.npy"))

    feature_dataset = FeatureSpaceDataset(features=src_features, labels=src_labels)

    proto_cfg = config.get("prototype", {})
    pt_dataset = prototype.run(
        dataset=feature_dataset,
        undersampling=proto_cfg.get("undersampling"),
        clustering=proto_cfg.get("clustering", "k_means"),
        k=proto_cfg.get("n_clusters", 1),
        seed=config["seed"],
    )

    ae_cfg = config.get("autoencoder", {})
    input_dim = feature_dataset.features_dim
    autoencoder = Autoencoder(
        arch=ae_cfg.get("arch", "variational_autoencoder"),
        input_dim=input_dim,
        hidden_dim=int(input_dim * ae_cfg.get("hidden_dim_ratio", 0.5)),
        latent_dim=int(input_dim * ae_cfg.get("latent_dim_ratio", 0.25)),
    )

    al_cfg = config.get("alignment", {})
    center_loss = CenterLoss(num_classes=2, feat_dim=input_dim, initial_centers=pt_dataset.ordered_centroids())

    lr = config["lr"]
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    es_cfg = config.get("early_stopping", {})
    es = EarlyStoppingConfig(patience=es_cfg.get("patience", 10))

    rl_cfg = config.get("reduce_lr")
    reduce_lr = None
    if rl_cfg:
        reduce_lr = ReduceLROnPlateauConfig(optimizer=ae_optimizer, patience=rl_cfg.get("patience", 10), factor=rl_cfg.get("factor", 0.1))

    ae_config = AutoencoderConfiguration(
        epochs=config["epochs"],
        optimizer=ae_optimizer,
        early_stopping=es,
        reduce_lr=reduce_lr,
        alignment_loss=center_loss,
        align_weight=al_cfg.get("align_weight", 0.9),
        kl_weight=al_cfg.get("kl_weight", 0.1),
        reconstruction_weight=al_cfg.get("reconstruction_weight", 0.0),
    )

    tgt_train_dataset = TargetDataset(split="train")
    tgt_train_data = utils.get_dataloader(tgt_train_dataset, batch_size=batch_size, shuffle=False)
    tgt_val_dataset = TargetDataset(split="val")
    tgt_val_data = utils.get_dataloader(tgt_val_dataset, batch_size=batch_size, shuffle=False)
    tgt_test_dataset = TargetDataset(split="test")
    tgt_test_data = utils.get_dataloader(tgt_test_dataset, batch_size=batch_size, shuffle=False)

    tgt_features_path = os.path.join(exp_dir, "target_features.npy")
    if not os.path.exists(tgt_features_path) or force:
        tgt_features, tgt_labels = extract_features(path=exp_dir, model=extractor, data=tgt_train_data, data_label="target")
        tgt_val_features, tgt_val_labels = extract_features(path=exp_dir, model=extractor, data=tgt_val_data, data_label="target_val")
        tgt_test_features, tgt_test_labels = extract_features(path=exp_dir, model=extractor, data=tgt_test_data, data_label="target_test")
    else:
        tgt_features = np.load(os.path.join(exp_dir, "target_features.npy"))
        tgt_labels = np.load(os.path.join(exp_dir, "target_labels.npy"))
        tgt_val_features = np.load(os.path.join(exp_dir, "target_val_features.npy"))
        tgt_val_labels = np.load(os.path.join(exp_dir, "target_val_labels.npy"))
        tgt_test_features = np.load(os.path.join(exp_dir, "target_test_features.npy"))
        tgt_test_labels = np.load(os.path.join(exp_dir, "target_test_labels.npy"))

    tgt_feature_dataset = FeatureSpaceDataset(features=tgt_features, labels=tgt_labels)
    tgt_feature_data = utils.get_dataloader(tgt_feature_dataset, batch_size=batch_size, shuffle=True)
    tgt_val_feature_dataset = FeatureSpaceDataset(features=tgt_val_features, labels=tgt_val_labels)
    tgt_val_feature_data = utils.get_dataloader(tgt_val_feature_dataset, batch_size=batch_size, shuffle=False)
    tgt_test_feature_dataset = FeatureSpaceDataset(features=tgt_test_features, labels=tgt_test_labels)
    tgt_test_feature_data = utils.get_dataloader(tgt_test_feature_dataset, batch_size=batch_size, shuffle=False)

    vae_exists = os.path.exists(os.path.join(exp_dir, "vae_encoder.pt"))
    if not vae_exists or force:
        stage2.train(path=exp_dir, model=autoencoder, train_data=tgt_feature_data, val_data=tgt_val_feature_data, config=ae_config)
    else:
        print(f"VAE weights found in {exp_dir}, skipping training.")
        autoencoder.load(exp_dir)

    aligned_features, aligned_labels = align_features(
        path=exp_dir, model=autoencoder, data=tgt_test_feature_data, data_label="target_aligned"
    )
    aligned_dataset = FeatureSpaceDataset(features=aligned_features, labels=aligned_labels)
    aligned_data = utils.get_dataloader(aligned_dataset, batch_size=batch_size, shuffle=False)

    source_pca, axis_limits = utils.plot_pca(
        path=exp_dir, features=src_features, labels=src_labels,
        title="Source Domain",
        prototypes=pt_dataset.features, prototype_labels=pt_dataset.labels,
    )
    utils.plot_pca(
        path=exp_dir, features=tgt_test_features, labels=tgt_test_labels,
        title="Target Domain (Source PCA)",
        pca=source_pca,
        prototypes=pt_dataset.features, prototype_labels=pt_dataset.labels,
        axis_limits=axis_limits,
    )
    utils.plot_pca(
        path=exp_dir, features=aligned_features, labels=aligned_labels,
        title="Aligned Target (Source PCA)",
        pca=source_pca,
        prototypes=pt_dataset.features, prototype_labels=pt_dataset.labels,
        axis_limits=axis_limits,
    )

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, test_precision, test_recall, test_f1 = stage2.test(
        path=stage1_exp_dir, model=classifier, data=tgt_test_feature_data, criterion=criterion
    )
    aligned_test_loss, aligned_test_acc, aligned_test_precision, aligned_test_recall, aligned_test_f1 = stage2.test(
        path=stage1_exp_dir, model=classifier, data=aligned_data, criterion=criterion
    )

    row = {
        "experiment": exp_dir,
        "stage": 2,
        "timestamp": datetime.now().isoformat(),
        "seed": config["seed"],
        "backbone": backbone,
        "unfrozen_layers": str(unfrozen_layers),
        "source_dataset": config["source_dataset"],
        "target_dataset": config["target_dataset"],
        "batch_size": batch_size,
        "lr": lr,
        "epochs": config["epochs"],
        "es_patience": es_cfg.get("patience", 10),
        "reduce_lr_patience": rl_cfg.get("patience", 10) if rl_cfg else "",
        "n_clusters": proto_cfg.get("n_clusters", 1),
        "clustering": proto_cfg.get("clustering", "k_means"),
        "undersampling": proto_cfg.get("undersampling", ""),
        "ae_arch": ae_cfg.get("arch", "variational_autoencoder"),
        "align_weight": al_cfg.get("align_weight", 0.9),
        "kl_weight": al_cfg.get("kl_weight", 0.1),
        "reconstruction_weight": al_cfg.get("reconstruction_weight", 0.0),
        "test_loss": round(test_loss, 6),
        "test_acc": round(test_acc, 6),
        "test_precision": round(test_precision, 6),
        "test_recall": round(test_recall, 6),
        "test_f1": round(test_f1, 6),
        "aligned_test_loss": round(aligned_test_loss, 6),
        "aligned_test_acc": round(aligned_test_acc, 6),
        "aligned_test_precision": round(aligned_test_precision, 6),
        "aligned_test_recall": round(aligned_test_recall, 6),
        "aligned_test_f1": round(aligned_test_f1, 6),
    }
    utils.append_results(row)
    print(f"\nResults saved to {utils.RESULTS_CSV}")


def run_experiment(config_path: str, force: bool):
    config_path = os.path.abspath(config_path)
    exp_dir = os.path.dirname(config_path)
    config = load_config(config_path)

    stage = config["stage"]
    print(f"\n{'='*60}")
    print(f"Experiment : {config_path}")
    print(f"Stage      : {stage}")
    print(f"{'='*60}")

    if stage == 1:
        run_stage1(config, exp_dir, force)
    elif stage == 2:
        run_stage2(config, exp_dir, force)
    else:
        raise ValueError(f"Unknown stage: {stage}")
