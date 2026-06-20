import os
import torch
import torch.nn as nn

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

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

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "skip_encoder.pt")
        torch.save(self.state_dict(), weight_path)

    def load(self, path):
        weight_path = os.path.join(path, "skip_encoder.pt")
        if not os.path.exists(weight_path):
            return
        self.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )

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

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "skip_decoder.pt")
        torch.save(self.state_dict(), weight_path)

    def load(self, path):
        weight_path = os.path.join(path, "skip_decoder.pt")
        if not os.path.exists(weight_path):
            return
        self.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )

class SkipAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super(SkipAutoencoder, self).__init__()
        self.encoder = _Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = _Decoder(input_dim, hidden_dim, latent_dim)
        self.skip_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat + self.skip_weight * x
        return x_hat, z

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.encoder.save(path)
        self.decoder.save(path)
        weight_path = os.path.join(path, "skip_weight.pt")
        torch.save(self.skip_weight, weight_path)

    def load(self, path):
        self.encoder.load(path)
        self.decoder.load(path)
        weight_path = os.path.join(path, "skip_weight.pt")
        if os.path.exists(weight_path):
            self.skip_weight = nn.Parameter(
                torch.load(weight_path, map_location=device, weights_only=True)
            )