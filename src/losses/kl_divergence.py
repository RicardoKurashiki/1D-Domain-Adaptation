import torch
import torch.nn as nn

class KLDivergenceLoss(nn.Module):
    def forward(self, mean, log_var):
        return -0.5 * torch.mean(1 + log_var - mean.pow(2) - log_var.exp())
