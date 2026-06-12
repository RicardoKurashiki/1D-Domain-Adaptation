import os
import torch
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

class KermanyDataset(Dataset):
    def __init__(self, transform=None, split="train"):
        self.split = split
        self.transform = transform
        if transform is None:
            self.transform = transforms.ToTensor()
        self.classes = {0: "NORMAL", 1: "PNEUMONIA"}
        self.n_classes = 2
        self.root = "./data/processed/kermany/"
        
        data = self.__getdataframe__(self.root)
        if split == "train":
            self.data = data[data["split"] == "train"]
            self.labels = self.data["label_idx"].tolist()
        elif split == "val":
            self.data = data[data["split"] == "val"]
            self.labels = self.data["label_idx"].tolist()
        elif split == "test":
            self.data = data[data["split"] == "test"]
            self.labels = self.data["label_idx"].tolist()
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["path"]
        label = self.data.iloc[idx]["label_idx"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, int(label)

    def __getdataframe__(self, path):
        data = []
        for spl in ["train", "val", "test"]:
            for cls in self.classes.keys():
                for img in os.listdir(os.path.join(path, spl, self.classes[cls])):
                    data.append({"split": spl, "label": self.classes[cls], "label_idx": cls, "path": os.path.join(path, spl, self.classes[cls], img)})
        return pd.DataFrame(data)