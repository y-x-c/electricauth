"""
PyTorch implementation of VAE found at: https://github.com/pytorch/examples/tree/master/vae.
"""

import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_length, n_sensor_channel,
        n_hidden = 400,
        n_latent_features=20, no_variational=False
    ):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_length*n_sensor_channel, n_hidden)
        self.fc21 = nn.Linear(n_hidden, n_latent_features)
        self.fc22 = nn.Linear(n_hidden, n_latent_features)
        self.fc3 = nn.Linear(n_latent_features, n_hidden)
        self.fc4 = nn.Linear(n_hidden, input_length * n_sensor_channel)
        
        self.no_variational = no_variational

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.no_variational:
            return mu
        else:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

