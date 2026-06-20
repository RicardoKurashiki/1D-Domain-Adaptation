import os
import torch
import torch.nn as nn

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

class _Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(_Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, latent_dim)

    def forward(self, x):
        # Dense(input_dim > latent_dim) = z
        return self.fc1(x)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "linear_encoder.pt")
        torch.save(self.state_dict(), weight_path)
    
    def load(self, path):
        weight_path = os.path.join(path, "linear_encoder.pt")
        if not os.path.exists(weight_path):
            return
        self.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )

class _Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(_Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, input_dim)

    def forward(self, z):
        # Dense(latent_dim > input_dim) = x_hat
        return self.fc1(z)
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "linear_decoder.pt")
        torch.save(self.state_dict(), weight_path)
    
    def load(self, path):
        weight_path = os.path.join(path, "linear_decoder.pt")
        if not os.path.exists(weight_path):
            return
        self.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )

class LinearAutoencoder(nn.Module):
    def __init__(self, input_dim:int, latent_dim:int):
        super(LinearAutoencoder, self).__init__()
        self.encoder = _Encoder(input_dim, latent_dim)
        self.decoder = _Decoder(input_dim, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.encoder.save(path)
        self.decoder.save(path)
    
    def load(self, path):
        self.encoder.load(path)
        self.decoder.load(path)