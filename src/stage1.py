import os
import yaml
import time
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from peft import LoraConfig

from .computational_metrics import ComputationalMetrics
from .configuration import Configuration, EarlyStoppingConfig, ReduceLROnPlateauConfig
from .models import ClassifierModel, ExperimentMetrics, FeatureExtractor, ClassificationHead
from .constants import DATASET_MAP
from .datasets import MiniBatchSampler
from .extract import extract_features
from . import utils

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

def train(path:str, model: ClassifierModel, train_data:DataLoader, val_data:DataLoader, config:Configuration, verbose:bool=True):
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
            running_loss=0.0
            running_corrects=0
            total_samples_processed=0

            pbar = tqdm(dataloaders[phase], desc=f"{phase.upper():5}", leave=False)
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == "train":
                    config.optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = config.criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        config.optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                total_samples_processed += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_acc = torch.sum(preds == labels.data).float() / inputs.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"})

            epoch_loss = running_loss/total_samples_processed
            epoch_acc = running_corrects.float()/total_samples_processed
            
            if verbose:
                print(f"{phase.upper():5} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

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

def test(path: str, model: ClassifierModel, data: DataLoader, criterion, verbose: bool = True):
    model.load(path)
    model = model.to(device)

    model.eval()

    running_loss=0.0
    total_samples=0

    all_preds=[]
    all_labels=[]

    pbar = tqdm(data, desc="TEST", leave=False)
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
        print(f"TEST  | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    return test_loss, float(test_acc), float(precision), float(recall), float(f1)

def run(config, exp_dir, force):
    seed = config["seed"]
    utils.set_seed(seed)
    
    metrics = ExperimentMetrics()
    metrics.exp_dir = exp_dir
    metrics.stage = 1
    metrics.seed = seed

    source_dataset = DATASET_MAP[config["source_dataset"]]
    metrics.source_dataset = config["source_dataset"]

    batch_size = config["batch_size"]
    metrics.batch_size = batch_size

    src_train = source_dataset(split="train", transform=utils.set_transformations(config, "train"))
    src_val   = source_dataset(split="val",   transform=utils.set_transformations(config, "val"))
    src_test  = source_dataset(split="test",  transform=utils.set_transformations(config, "test"))
    
    sampler = MiniBatchSampler(src_train, batch_size)

    src_train_data = utils.get_dataloader(src_train, sampler=sampler, shuffle=False)
    src_val_data = utils.get_dataloader(src_val, batch_size=batch_size, shuffle=False)
    src_test_data = utils.get_dataloader(src_test, batch_size=batch_size, shuffle=False)
    
    backbone = config["model"].get("backbone", "resnet18")
    metrics.backbone = backbone

    unfrozen_layers = config["model"].get("unfrozen_layers", None)
    metrics.unfrozen_layers = str(unfrozen_layers)

    lora_config = None
    if backbone == "vit_lora":
        lora_cfg = config["model"].get("lora", {})
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 8),
            lora_alpha=lora_cfg.get("alpha", 16),
            lora_dropout=lora_cfg.get("dropout", 0.1),
            target_modules=lora_cfg.get("target_modules", ["q_proj", "v_proj"]),
            bias=lora_cfg.get("bias", "none"),
        )

    extractor = FeatureExtractor(backbone=backbone, unfrozen_layers=unfrozen_layers, lora_config=lora_config)
    classifier = ClassificationHead(in_features=extractor.num_ftrs, out_features=src_train.n_classes)
    model = ClassifierModel(extractor, classifier)

    criterion = nn.CrossEntropyLoss()
    
    lr = config["lr"]
    metrics.lr = lr
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    es_cfg = config.get("early_stopping", {})
    es = EarlyStoppingConfig(patience=es_cfg.get("patience", 10), start_epoch=es_cfg.get("start_epoch", 0))
    metrics.es_patience = es_cfg.get("patience", 10)

    rl_cfg = config.get("reduce_lr", {})
    reduce_lr = ReduceLROnPlateauConfig(optimizer=optimizer, patience=rl_cfg.get("patience", 10), factor=rl_cfg.get("factor", 0.1))
    metrics.reduce_lr_patience = rl_cfg.get("patience", 10)

    epochs = config["epochs"]
    metrics.epochs = epochs

    training_config = Configuration(
        epochs=epochs,
        optimizer=optimizer,
        criterion=criterion,
        early_stopping=es,
        reduce_lr=reduce_lr,
    )

    comp_metrics = ComputationalMetrics(
        trainable_params=model.get_trainable_params(),
        model_size=model.get_model_size(),
    )

    weights_exist = os.path.exists(os.path.join(exp_dir, "extractor_weight.pt"))
    if not weights_exist or force:
        comp_metrics.start()
        train(path=exp_dir, model=model, train_data=src_train_data, val_data=src_val_data, config=training_config)
        comp_metrics.finish()
    else:
        print(f"Weights found in {exp_dir}, skipping training.")
        model.load(exp_dir)

    metrics.set_computational_metrics(comp_metrics)

    source_loss, source_acc, source_pre, source_rec, source_f1 = test(
        path=exp_dir, model=model, data=src_test_data, criterion=criterion
    )

    metrics.source_loss = round(source_loss, 6)
    metrics.source_acc = round(source_acc, 6)
    metrics.source_pre = round(source_pre, 6)
    metrics.source_rec = round(source_rec, 6)
    metrics.source_f1 = round(source_f1, 6)

    src_train_feat = source_dataset(split="train", transform=utils.set_transformations(config, "test"))
    src_train_feat_data = utils.get_dataloader(src_train_feat, batch_size=batch_size, shuffle=False)
    if not os.path.exists(os.path.join(exp_dir, "source_features.npy")) or force:
        extract_features(path=exp_dir, model=extractor, data=src_train_feat_data, data_label="source")

    src_test_feat = source_dataset(split="test", transform=utils.set_transformations(config, "test"))
    src_test_feat_data = utils.get_dataloader(src_test_feat, batch_size=batch_size, shuffle=False)
    if not os.path.exists(os.path.join(exp_dir, "source_test_features.npy")) or force:
        extract_features(path=exp_dir, model=extractor, data=src_test_feat_data, data_label="source_test")

    targets = config.get("targets")
    if not targets:
        targets = [config["target_dataset"]] if config.get("target_dataset") else []

    if not targets:
        metrics.exp_dir = exp_dir
        utils.save_results(metrics.to_json(), exp_dir)
        return

    for target_name in targets:
        target_dataset = DATASET_MAP[target_name]
        target_dir = os.path.join(exp_dir, target_name)

        tgt_train = target_dataset(split="train", transform=utils.set_transformations(config, "test"))
        tgt_val = target_dataset(split="val", transform=utils.set_transformations(config, "test"))
        tgt_test = target_dataset(split="test", transform=utils.set_transformations(config, "test"))
        tgt_train_data = utils.get_dataloader(tgt_train, batch_size=batch_size, shuffle=False)
        tgt_val_data = utils.get_dataloader(tgt_val, batch_size=batch_size, shuffle=False)
        tgt_test_data = utils.get_dataloader(tgt_test, batch_size=batch_size, shuffle=False)

        if not os.path.exists(os.path.join(target_dir, "target_features.npy")) or force:
            extract_features(path=target_dir, model=extractor, data=tgt_train_data, data_label="target", load_path=exp_dir)
            extract_features(path=target_dir, model=extractor, data=tgt_val_data, data_label="target_val", load_path=exp_dir)
            extract_features(path=target_dir, model=extractor, data=tgt_test_data, data_label="target_test", load_path=exp_dir)

        target_loss, target_acc, target_pre, target_rec, target_f1 = test(
            path=exp_dir, model=model, data=tgt_test_data, criterion=criterion
        )

        metrics.exp_dir = target_dir
        metrics.target_dataset = target_name
        metrics.target_loss = round(target_loss, 6)
        metrics.target_acc = round(target_acc, 6)
        metrics.target_pre = round(target_pre, 6)
        metrics.target_rec = round(target_rec, 6)
        metrics.target_f1 = round(target_f1, 6)

        utils.save_results(metrics.to_json(), target_dir)