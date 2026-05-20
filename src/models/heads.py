import os
import torch
import torch.nn as nn

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

class ClassificationHead(nn.Module):
    def __init__(self, in_features:int, out_features:int, dropout=0.2):
        super(ClassificationHead, self).__init__()
        self.out_features = out_features
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, out_features),
        )

    def forward(self, x):
        return self.classifier(x)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "head_weight.pt")
        torch.save(self.classifier.state_dict(), weight_path)
    
    def load(self, path):
        weight_path = os.path.join(path, "head_weight.pt")
        if not os.path.exists(weight_path):
            return
        self.classifier.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )