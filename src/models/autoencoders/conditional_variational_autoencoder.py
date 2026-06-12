import os
import torch
import torch.nn as nn

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

class _Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_classes):
        super(_Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim+n_classes, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x, c):
        x_c = torch.cat([x,c], dim=1)
        h_ = self.LeakyReLU(self.FC_input(x_c))
        h_ = self.LeakyReLU(self.FC_input2(h_))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        return mean, log_var

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "cvae_encoder.pt")
        torch.save(self.state_dict(), weight_path)
    
    def load(self, path):
        weight_path = os.path.join(path, "cvae_encoder.pt")
        if not os.path.exists(weight_path):
            return
        self.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )

class _Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim, n_classes):
        super(_Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim+n_classes, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)

    def forward(self, x, c):
        x_c = torch.cat([x,c], dim=1)
        h = self.LeakyReLU(self.FC_hidden(x_c))
        h = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = self.FC_output(h)
        return x_hat

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        weight_path = os.path.join(path, "cvae_decoder.pt")
        torch.save(self.state_dict(), weight_path)
    
    def load(self, path):
        weight_path = os.path.join(path, "cvae_decoder.pt")
        if not os.path.exists(weight_path):
            return
        self.load_state_dict(
            torch.load(weight_path, map_location=device, weights_only=True)
        )


class ConditionalVariationalAutoencoder(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=128, latent_dim=64, n_classes=2):
        super(ConditionalVariationalAutoencoder, self).__init__()
        self.encoder = _Encoder(input_dim, hidden_dim, latent_dim, n_classes)
        self.decoder = _Decoder(input_dim, hidden_dim, latent_dim, n_classes)

    def reparameterization(self, mean, std):
        epsilon = torch.randn_like(std)
        z = mean + std * epsilon
        return z

    def forward(self, x, c):
        mean, log_var = self.encoder(x, c)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var))
        x_hat = self.decoder(z, c)
        return x_hat, z, mean, log_var

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.encoder.save(path)
        self.decoder.save(path)
    
    def load(self, path):
        self.encoder.load(path)
        self.decoder.load(path)