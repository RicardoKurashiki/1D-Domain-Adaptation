import os
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, arch:str, input_dim:int, hidden_dim:int, latent_dim:int):
        super(Autoencoder, self).__init__()
        self.arch = arch
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.model = self.__get_model__(arch)

    def __get_model__(self, arch:str):
        if arch == "simple_autoencoder":
            from .autoencoders import SimpleAutoencoder
            return SimpleAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)
        if arch == "variational_autoencoder":
            from .autoencoders import VariationalAutoencoder
            return VariationalAutoencoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)

    def forward(self, x):
        return self.model(x)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save(path)

    def load(self, path):
        self.model.load(path)

    def get_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def get_model_size(self):
        param_size = 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return (param_size + buffer_size) / 1024**2
