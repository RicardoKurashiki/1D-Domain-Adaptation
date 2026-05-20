import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .configuration import Configuration
from .computational_metrics import ComputationalMetrics
from .models import ClassifierModel

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
                current_lr = config.reduce_lr.scheduler.config.optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch + 1}/{config.epochs} - New Learning Rate: {current_lr:.6f}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    
    model.load(path)
    return True

