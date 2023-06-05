# -*- coding: utf-8 -*-
"""
Variational Autoencoder
Beta Variational Autoencoder, where the beta parameter adjusts the KL-divergency
loss.
"""
import torch
from torch import nn


class VAE(nn.Module):
    def __init__(self,
                 feature_dimension,
                 latent_dimension):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(feature_dimension, 100),
            self._block(100, 150),
            nn.Linear(150, latent_dimension)
        )
        self.z_mean = nn.Linear(latent_dimension, latent_dimension)
        self.z_log_var = nn.Linear(latent_dimension, latent_dimension)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dimension, 150),
            self._block(150, 100),
            nn.Linear(100, feature_dimension),
            nn.Sigmoid()
        )

    def _block(self, input_d, n_nodes):
        return nn.Sequential(
            nn.Linear(in_features=input_d, out_features=n_nodes),
            nn.BatchNorm1d(n_nodes),
            nn.LeakyReLU(0.2))

    def encode(self, x):
        x = self.encoder(x)
        mu = self.z_mean(x)
        log_var = self.z_log_var(x)
        return mu, log_var

    def get_latent(self, x):
        latent = self.encoder(x)
        return latent

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        epsilon = torch.randn_like(log_var)
        z_parametrised = epsilon * (torch.exp(log_var / 2)) + mu
        x = self.decode(z_parametrised)
        return [x, mu, log_var]
