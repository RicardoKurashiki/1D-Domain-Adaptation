import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

from .configuration import AutoencoderConfiguration
from .models import ClassificationHead, Autoencoder
from .losses import KLDivergenceLoss

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

kl_loss_fn = KLDivergenceLoss()

def train(path:str, model:Autoencoder, train_data:DataLoader, val_data:DataLoader, config:AutoencoderConfiguration, verbose:bool=True):
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

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model.load(path)
    return True

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
