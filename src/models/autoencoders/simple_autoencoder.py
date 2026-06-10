import torch
import torch.nn as nn

class _Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        z = self.fc2(x)
        return z

class _Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(_Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, z):
        z = self.fc1(z)
        z = self.relu(z)
        x_hat = self.fc2(z)
        return x_hat


class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, latent_dim:int):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = _Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = _Decoder(input_dim, hidden_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
