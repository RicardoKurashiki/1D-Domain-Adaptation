import os
import yaml
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from .computational_metrics import ComputationalMetrics
from .configuration import AutoencoderConfiguration, EarlyStoppingConfig, ReduceLROnPlateauConfig
from .models import ClassificationHead, Autoencoder, ExperimentMetrics
from .losses import KLDivergenceLoss, CenterLoss
from .datasets import FeatureSpaceDataset
from .extract import align_features
from . import utils, prototype

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def train(path:str, model:Autoencoder, train_data:DataLoader, val_data:DataLoader, config:AutoencoderConfiguration, verbose:bool=True):
    kl_loss_fn = KLDivergenceLoss()
    model = model.to(device)

    dataloaders = {
        "train": train_data,
        "val": val_data
    }

    best_val_loss = float("inf")

    for epoch in range(config.epochs):
        if verbose:
            print(f"\nEpoch {epoch+1} / {config.epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            total_samples_processed = 0

            pbar = tqdm(dataloaders[phase], desc=f"{phase.upper():5}", leave=False)
            for inputs, labels in pbar:
                inputs = inputs.float().to(device)
                labels = labels.long().to(device)

                if phase == "train":
                    config.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    is_vae = model.arch == "variational_autoencoder"
                    if is_vae:
                        x_recon, z, mean, log_var = model(inputs)
                        kl = kl_loss_fn(mean, log_var)
                    else:
                        x_recon, z = model(inputs)
                        kl = torch.tensor(0.0, device=device)

                    rec_loss = config.reconstruction_loss(x_recon, inputs)
                    align_loss = config.alignment_loss(x_recon, labels)
                    loss = (rec_loss * config.reconstruction_weight) + (align_loss * config.align_weight) + (kl * config.kl_weight)

                    if phase == "train":
                        loss.backward()
                        config.optimizer.step()

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                running_loss += loss.item() * inputs.size(0)
                total_samples_processed += inputs.size(0)
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "align": f"{align_loss.item():.4f}",
                    "kl": f"{kl.item():.4f}",
                    "rec": f"{rec_loss.item():.4f}",
                })

            epoch_loss = running_loss / total_samples_processed

            if verbose:
                print(f"{phase.upper():5} | Loss: {epoch_loss:.4f}")

            if phase == "val":
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    model.save(path)
                    if verbose:
                        print(f"Model saved (best loss: {best_val_loss:.4f})")
                if config.has_early_stopping:
                    config.early_stopping.check(epoch_loss)
                    if config.early_stopping.early_stop:
                        break

        if config.has_early_stopping and config.early_stopping.early_stop:
            break

        if config.has_reduce_lr:
            config.reduce_lr.scheduler.step(epoch_loss)
            if verbose:
                current_lr = config.reduce_lr.get_last_lr()
                print(f"Epoch {epoch + 1}/{config.epochs} - New Learning Rate: {current_lr:.6f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.load(path)
    return True

def _load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def test(path: str, model:ClassificationHead, data:DataLoader, criterion, verbose: bool = True):
    model.load(path)
    model = model.to(device)
    model.eval()
    running_loss=0.0
    total_samples=0
    all_preds=[]
    all_labels=[]
    pbar = tqdm(data, desc="FEATURE TEST", leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            total_samples += inputs.size(0)
            running_loss += loss.item() * inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    test_loss = running_loss/total_samples
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    test_acc = (all_preds == all_labels).sum() / len(all_labels)
    precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

    if verbose:
        print(f"FEATURE TEST | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    return test_loss, float(test_acc), float(precision), float(recall), float(f1)

def run(config: dict, exp_dir: str, force: bool):
    utils.set_seed(config["seed"])

    metrics = ExperimentMetrics()
    metrics.exp_dir = exp_dir
    metrics.stage = 2
    metrics.seed = config["seed"]
    metrics.source_dataset = config["source_dataset"]
    metrics.target_dataset = config["target_dataset"]

    batch_size = config["batch_size"]
    metrics.batch_size = batch_size

    stage1_config_path = config["stage1_config"]
    stage1_cfg = _load_config(stage1_config_path)
    stage1_exp_dir = config.get("stage1_dir") or os.path.dirname(os.path.abspath(stage1_config_path))

    metrics.backbone = stage1_cfg["model"]["backbone"]
    metrics.unfrozen_layers = str(stage1_cfg["model"].get("unfrozen_layers"))

    stage1_target_dir = os.path.join(stage1_exp_dir, config["target_dataset"])
    if not os.path.exists(os.path.join(stage1_target_dir, "target_features.npy")):
        print(f"Target features not found in {stage1_target_dir}, skipping experiment.")
        return

    src_features = np.load(os.path.join(stage1_exp_dir, "source_features.npy"))
    src_labels = np.load(os.path.join(stage1_exp_dir, "source_labels.npy"))
    src_test_features = np.load(os.path.join(stage1_exp_dir, "source_test_features.npy"))
    src_test_labels = np.load(os.path.join(stage1_exp_dir, "source_test_labels.npy"))

    tgt_features = np.load(os.path.join(stage1_target_dir, "target_features.npy"))
    tgt_labels = np.load(os.path.join(stage1_target_dir, "target_labels.npy"))
    tgt_val_features = np.load(os.path.join(stage1_target_dir, "target_val_features.npy"))
    tgt_val_labels = np.load(os.path.join(stage1_target_dir, "target_val_labels.npy"))
    tgt_test_features = np.load(os.path.join(stage1_target_dir, "target_test_features.npy"))
    tgt_test_labels = np.load(os.path.join(stage1_target_dir, "target_test_labels.npy"))

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
        arch=ae_cfg.get("arch", "simple_autoencoder"),
        input_dim=input_dim,
        hidden_dim=int(input_dim * ae_cfg.get("hidden_dim_ratio", 0.5)),
        latent_dim=int(input_dim * ae_cfg.get("latent_dim_ratio", 0.25)),
    )

    al_cfg = config.get("alignment", {})
    center_loss = CenterLoss(num_classes=2, feat_dim=input_dim, initial_centers=pt_dataset.ordered_centroids())

    lr = config["lr"]
    metrics.lr = lr
    ae_optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    es_cfg = config.get("early_stopping", {})
    es = EarlyStoppingConfig(patience=es_cfg.get("patience", 10))
    metrics.es_patience = es_cfg.get("patience", 10)

    rl_cfg = config.get("reduce_lr")
    reduce_lr = None
    if rl_cfg:
        reduce_lr = ReduceLROnPlateauConfig(optimizer=ae_optimizer, patience=rl_cfg.get("patience", 10), factor=rl_cfg.get("factor", 0.1))

    metrics.reduce_lr_patience = rl_cfg.get("patience", 10) if rl_cfg else ""
    metrics.epochs = config["epochs"]
    metrics.n_clusters = proto_cfg.get("n_clusters", 1)
    metrics.clustering = proto_cfg.get("clustering", "k_means")
    metrics.undersampling = proto_cfg.get("undersampling", "")
    metrics.ae_arch = ae_cfg.get("arch", "variational_autoencoder")
    metrics.align_weight = al_cfg.get("align_weight", 0.9)
    metrics.kl_weight = al_cfg.get("kl_weight", 0.1)
    metrics.reconstruction_weight = al_cfg.get("reconstruction_weight", 0.0)

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

    classifier = ClassificationHead(in_features=input_dim, out_features=2)

    tgt_feature_dataset = FeatureSpaceDataset(features=tgt_features, labels=tgt_labels)
    tgt_feature_data = utils.get_dataloader(tgt_feature_dataset, batch_size=batch_size, shuffle=True)
    tgt_val_feature_dataset = FeatureSpaceDataset(features=tgt_val_features, labels=tgt_val_labels)
    tgt_val_feature_data = utils.get_dataloader(tgt_val_feature_dataset, batch_size=batch_size, shuffle=False)
    tgt_test_feature_dataset = FeatureSpaceDataset(features=tgt_test_features, labels=tgt_test_labels)
    tgt_test_feature_data = utils.get_dataloader(tgt_test_feature_dataset, batch_size=batch_size, shuffle=False)

    src_test_feature_dataset = FeatureSpaceDataset(features=src_test_features, labels=src_test_labels)
    src_test_feature_data = utils.get_dataloader(src_test_feature_dataset, batch_size=batch_size, shuffle=False)

    comp_metrics = ComputationalMetrics(
        trainable_params=autoencoder.get_trainable_params(),
        model_size=autoencoder.get_model_size(),
    )

    vae_exists = os.path.exists(os.path.join(exp_dir, "vae_encoder.pt"))
    if not vae_exists or force:
        comp_metrics.start()
        train(path=exp_dir, model=autoencoder, train_data=tgt_feature_data, val_data=tgt_val_feature_data, config=ae_config)
        comp_metrics.finish()
    else:
        print(f"VAE weights found in {exp_dir}, skipping training.")
        autoencoder.load(exp_dir)

    metrics.set_computational_metrics(comp_metrics)

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
    source_loss, source_acc, source_pre, source_rec, source_f1 = test(
        path=stage1_exp_dir, model=classifier, data=src_test_feature_data, criterion=criterion
    )
    target_loss, target_acc, target_pre, target_rec, target_f1 = test(
        path=stage1_exp_dir, model=classifier, data=tgt_test_feature_data, criterion=criterion
    )
    aligned_loss, aligned_acc, aligned_pre, aligned_rec, aligned_f1 = test(
        path=stage1_exp_dir, model=classifier, data=aligned_data, criterion=criterion
    )

    metrics.exp_dir = exp_dir
    metrics.source_loss = round(source_loss, 6)
    metrics.source_acc = round(source_acc, 6)
    metrics.source_pre = round(source_pre, 6)
    metrics.source_rec = round(source_rec, 6)
    metrics.source_f1 = round(source_f1, 6)
    metrics.target_loss = round(target_loss, 6)
    metrics.target_acc = round(target_acc, 6)
    metrics.target_pre = round(target_pre, 6)
    metrics.target_rec = round(target_rec, 6)
    metrics.target_f1 = round(target_f1, 6)
    metrics.aligned_loss = round(aligned_loss, 6)
    metrics.aligned_acc = round(aligned_acc, 6)
    metrics.aligned_pre = round(aligned_pre, 6)
    metrics.aligned_rec = round(aligned_rec, 6)
    metrics.aligned_f1 = round(aligned_f1, 6)

    utils.save_results(metrics.to_json(), exp_dir)
