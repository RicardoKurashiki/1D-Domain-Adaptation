import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from configuration import Configuration
from computational_metrics import ComputationalMetrics

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

def train(path:str, model: nn.Module, dataset:Dataset, config:Configuration):
    model = model.to(device)
    metrics = ComputationalMetrics(trainable_params=model.get_trainable_params(), model_size=model.get_model_size())
    metrics.start()
    for epoch in config.epochs:
        if config.has_early_stopping and config.early_stopping.early_stop:
            break
        print(f"Epoch {epoch+1} / {config.epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()
            running_loss=0.0
            running_corrects=0
            total_samples_processed=0
            for inputs, labels in pbar:
                batch_start = time.time()
                inputs = inputs.to(device)
                labels = labels.to(device)

                if phase == "train":
                    optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                total_samples_processed += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_acc = torch.sum(preds == labels.data).float() / inputs.size(0)
                pbar.set_postfix(
                    {"loss": f"{loss.item():.4f}", "acc": f"{batch_acc:.4f}"}
                )
        