import os
import torch
import torch.nn as nn

class ClassifierModel(nn.Module):
    def __init__(self, extractor: nn.Module, head: nn.Module):
        super(ClassifierModel, self).__init__()
        self.extractor = extractor
        self.head=head
    
    def forward(self, x):
        z = self.extractor(x)
        c = self.head(z)
        return c

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.extractor.save(path)
        self.head.save(path)

    def load(self, path):
        os.makedirs(path, exist_ok=True)
        self.extractor.load(path)
        self.head.load(path)
    
    def get_trainable_params(self):
        return self.extractor.get_trainable_params() + self.head.get_trainable_params()

    def get_model_size(self):
        return self.extractor.get_model_size() + self.head.get_model_size()