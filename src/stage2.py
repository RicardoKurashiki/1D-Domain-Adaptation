import torch
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from .models import ClassificationHead

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

def test_features(path: str, model: ClassificationHead, data: DataLoader, criterion, verbose: bool = True):
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

    print(test_acc)