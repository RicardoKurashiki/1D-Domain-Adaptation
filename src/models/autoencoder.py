import os
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, arch:str, input_dim:int, hidden_dim:int, latent_dim:int, n_classes:int):
        super(Autoencoder, self).__init__()
        self.arch = arch
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.model = self.__get_model__(arch)
    
    def __get_model__(self, arch:str):
        if arch == "simple_autoencoder":
            from .autoencoders import SimpleAutoencoder
            return SimpleAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        if arch == "variational_autoencoder":
            from .autoencoders import VariationalAutoencoder
            return VariationalAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        if arch == "conditional_variational_autoencoder":
            from .autoencoders import ConditionalVariationalAutoencoder
            return ConditionalVariationalAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim, n_classes=self.n_classes)
        
    def forward(self, x, labels=None):
        if self.arch == "conditional_variational_autoencoder":
            if labels is None:
                raise ValueError("labels must be provided for conditional_variational_autoencoder")
            import torch.nn.functional as F
            c = F.one_hot(labels, num_classes=self.n_classes).float()
            return self.model(x, c)
        return self.model(x)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(path)
    
    def load(self, path):
        self.model.load(path)
